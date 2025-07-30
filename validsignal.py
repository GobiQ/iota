def analyze_rsi_signals(signal_prices, target_prices, rsi_threshold, comparison="less_than"):
    """Analyze RSI signals for a specific threshold"""
    # Calculate RSI for the SIGNAL ticker
    signal_rsi = calculate_rsi(signal_prices)
    
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
            'trades': []
        }
    
    returns = np.array([trade['return'] for trade in trades])
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
    sortino_ratio = calculate_sortino_ratio(returns)
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'returns': returns,
        'avg_hold_days': avg_hold_days,
        'sortino_ratio': sortino_ratio,
        'equity_curve': equity_curve,
        'trades': trades
    }                    # INDIVIDUAL STRATEGY ANALYSIS
                    st.subheader("Individual Strategy Analysis")
                    
                    # Strategy selection
                    strategy_type = st.selectbox(
                        "Choose strategy to analyze:",
                        ["Best Cumulative Return", "Best Sortino Ratio", "Custom Selection"],
                        help="Select which strategy's equity curve to display in detail"
                    )
                    
                    if strategy_type == "Best Cumulative Return":
                        selected_idx = best_return_idx
                        selected_threshold = best_return_threshold
                        selected_name = "Best Cumulative Return"
                    elif strategy_type == "Best Sortino Ratio" and not valid_sortino_df.empty:
                        selected_idx = best_sortino_idx
                        selected_threshold = best_sortino_threshold  
                        selected_name = "Best Sortino Ratio"
                    else:  # Custom selection
                        available_thresholds = results_df[results_df['Total_Trades'] > 0]['RSI_Threshold'].tolist()
                        if available_thresholds:
                            selected_threshold = st.selectbox(
                                "Select RSI Threshold:",
                                options=available_thresholds,
                                format_func=lambda x: f"RSI {comparison.replace('_', ' ').title()} {x}"
                            )
                            selected_idx = results_df[results_df['RSI_Threshold'] == selected_threshold].index[0]
                            selected_name = f"RSI {selected_threshold}"
                        else:
                            st.error("No strategies with trades available")
                            selected_idx = None
                    
                    if selected_idx is not None:
                        # Get data for selected strategy
                        selected_data = results_df.loc[selected_idx]
                        selected_equity_curve = selected_data['equity_curve']
                        selected_trades = selected_data['trades']
                        
                        # Show metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{selected_data['Total_Return']:.2%}")
                        with col2:
                            st.metric("Sortino Ratio", f"{selected_data['Sortino_Ratio']:.2f}" if not np.isinf(selected_data['Sortino_Ratio']) else "∞")
                        with col3:
                            st.metric("Win Rate", f"{selected_data['Win_Rate']:.1%}")
                        with col4:
                            st.metric("Total Trades", selected_data['Total_Trades'])
                        
                        # Create detailed equity curve
                        if selected_equity_curve is not None:
                            fig_detailed = go.Figure()
                            
                            # Add strategy curve
                            fig_detailed.add_trace(go.Scatter(
                                x=selected_equity_curve.index,
                                y=selected_equity_curve.values,
                                mode='lines',
                                name=f'{selected_name}',
                                line=dict(color='blue', width=3)
                            ))
                            
                            # Add SPY benchmark for comparison
                            fig_detailed.add_trace(go.Scatter(
                                x=spy_benchmark.index,
                                y=spy_benchmark.values,
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
                                
                                fig_detailed.add_trace(go.Scatter(
                                    x=entry_dates,
                                    y=entry_values,
                                    mode='markers',
                                    name='Buy',
                                    marker=dict(color='green', size=8, symbol='triangle-up')
                                ))
                                
                                fig_detailed.add_trace(go.Scatter(
                                    x=exit_dates,
                                    y=exit_values,
                                    mode='markers',
                                    name='Sell',
                                    marker=dict(color='red', size=8, symbol='triangle-down')
                                ))
                            
                            fig_detailed.update_layout(
                                title=f"Detailed Analysis: {selected_name} vs SPY Buy & Hold",
                                xaxis_title="Date",
                                yaxis_title="Equity Value",
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            
                            st.plotly_chart(fig_detailed, use_container_width=True)
                            
                            # Trade details
                            if selected_trades:
                                with st.expander(f"Trade Details - {selected_name}"):
                                    trades_df = pd.DataFrame(selected_trades)
                                    trades_df['return'] = trades_df['return'].apply(lambda x: f"{x:.2%}")
                                    trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
                                    trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
                                    st.dataframe(trades_df, use_container_width=True)import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.title("RSI Signal Analysis Engine")
st.write("Analyze RSI signals across different thresholds for any ticker pair")

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
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
    """Calculate RSI with standard 14-period window"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi(prices, window=14):
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

def get_stock_data(ticker, start_date=None, end_date=None):
    """Analyze RSI signals for a specific threshold"""
    rsi = calculate_rsi(signal_prices)
    
    # Generate buy signals based on RSI threshold and comparison
    if comparison == "less_than":
        # RSI ≤ threshold: signal is ON when RSI is at or below threshold
        signals = (rsi <= rsi_threshold).astype(int)
    else:  # greater_than
        # RSI ≥ threshold: signal is ON when RSI is at or above threshold
        signals = (rsi >= rsi_threshold).astype(int)
    
    # Calculate equity curve day by day
    equity_curve = pd.Series(1.0, index=target_prices.index)
    current_equity = 1.0
    in_position = False
    entry_equity = 1.0
    entry_date = None
    entry_price = None
    trades = []
    
    for i, date in enumerate(target_prices.index):
        current_signal = signals[date] if date in signals.index else 0
        current_price = target_prices[date]
        
        if current_signal == 1 and not in_position:
            # Enter position - buy at close
            in_position = True
            entry_equity = current_equity
            entry_date = date
            entry_price = current_price
            
        elif current_signal == 0 and in_position:
            # Exit position - sell at close
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
            # Mark-to-market the position
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
            'equity_curve': equity_curve,
            'trades': []
        }
    
    returns = np.array([trade['return'] for trade in trades])
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    avg_hold_days = np.mean([trade['hold_days'] for trade in trades])
    sortino_ratio = calculate_sortino_ratio(returns)
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'returns': returns,
        'avg_hold_days': avg_hold_days,
        'sortino_ratio': sortino_ratio,
        'equity_curve': equity_curve,
        'trades': trades
    }

def run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, comparison, start_date=None, end_date=None):
    """Run comprehensive RSI analysis across the specified range"""
    
    # Fetch data
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker, start_date, end_date)
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker, start_date, end_date)
    
    # Fetch SPY data for benchmark comparison
    with st.spinner("Fetching SPY benchmark data..."):
        spy_data = get_stock_data("SPY", start_date, end_date)
    
    if signal_data is None or target_data is None or spy_data is None:
        return None, None
    
    # Align data on common dates
    # Note: signal_data is still used for compatibility but RSI is calculated on target_data in analyze_rsi_signals
    common_dates = signal_data.index.intersection(target_data.index).intersection(spy_data.index)
    signal_data = signal_data[common_dates]
    target_data = target_data[common_dates]
    spy_data = spy_data[common_dates]
    
    # Create SPY buy-and-hold benchmark
    spy_benchmark = spy_data / spy_data.iloc[0]  # Normalize to start at 1.0
    
    # Generate RSI thresholds (every 0.5)
    rsi_thresholds = np.arange(rsi_min, rsi_max + 0.5, 0.5)
    
    results = []
    
    progress_bar = st.progress(0)
    total_thresholds = len(rsi_thresholds)
    
    for i, threshold in enumerate(rsi_thresholds):
        analysis = analyze_rsi_signals(signal_data, target_data, threshold, comparison)
        
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
            'equity_curve': analysis['equity_curve'],
            'trades': analysis['trades']
        })
        
        progress_bar.progress((i + 1) / total_thresholds)
    
    return pd.DataFrame(results), spy_benchmark

# Streamlit Interface
st.sidebar.header("Configuration")

# Input fields
signal_ticker = st.sidebar.text_input("Signal Ticker", value="SPY", help="Ticker to generate RSI signals from (currently not used - RSI calculated on target)")
target_ticker = st.sidebar.text_input("Target Ticker", value="QQQ", help="Ticker to buy based on its own RSI signals")

# Date range selection
st.sidebar.subheader("Date Range")
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
st.sidebar.subheader("RSI Configuration")
comparison = st.sidebar.selectbox("RSI Condition", 
                                 ["less_than", "greater_than"], 
                                 format_func=lambda x: "RSI ≤ threshold" if x == "less_than" else "RSI ≥ threshold",
                                 help="Buy when RSI is less than or greater than threshold")

if comparison == "less_than":
    default_min, default_max = 20, 40
    st.sidebar.write("Buy signals: Target RSI ≤ threshold")
else:
    default_min, default_max = 60, 80
    st.sidebar.write("Buy signals: Target RSI ≥ threshold")

rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5)
rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5)

if rsi_min >= rsi_max:
    st.sidebar.error("RSI Min must be less than RSI Max")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analysis Configuration")
    st.write(f"**Signal Ticker:** {signal_ticker} (not currently used)")
    st.write(f"**Target Ticker:** {target_ticker}")
    st.write(f"**RSI Condition:** {target_ticker} RSI {'≤' if comparison == 'less_than' else '≥'} threshold")
    st.write(f"**RSI Range:** {rsi_min} - {rsi_max}")
    if use_date_range and start_date and end_date:
        st.write(f"**Date Range:** {start_date} to {end_date}")
    else:
        st.write(f"**Date Range:** Maximum available data")

with col2:
    st.subheader("Strategy Logic")
    if comparison == "less_than":
        st.info(f"🔵 BUY {target_ticker} at close when {target_ticker} RSI ≤ threshold\n\n📈 SELL {target_ticker} at close when {target_ticker} RSI > threshold")
    else:
        st.info(f"🔵 BUY {target_ticker} at close when {target_ticker} RSI ≥ threshold\n\n📈 SELL {target_ticker} at close when {target_ticker} RSI < threshold")

if st.button("Run RSI Analysis", type="primary"):
    if rsi_min < rsi_max and (not use_date_range or (start_date and end_date and start_date < end_date)):
        try:
            results_df, spy_benchmark = run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, comparison, start_date, end_date)
            
            if results_df is not None and spy_benchmark is not None and not results_df.empty:
                st.success("✓ Analysis completed successfully!")
                
                # Display results table
                st.subheader("RSI Analysis Results")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}")
                display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Sortino_Ratio'] = display_df['Sortino_Ratio'].apply(lambda x: f"{x:.2f}" if not np.isinf(x) else "∞")
                display_df['Avg_Hold_Days'] = display_df['Avg_Hold_Days'].apply(lambda x: f"{x:.1f}")
                display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.2%}")
                display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Final_Equity'] = display_df['Final_Equity'].apply(lambda x: f"{x:.3f}")
                
                # Drop the equity_curve and trades columns for display
                display_cols = ['RSI_Threshold', 'Total_Trades', 'Win_Rate', 'Avg_Return', 
                               'Total_Return', 'Sortino_Ratio', 'Final_Equity', 'Avg_Hold_Days', 'Return_Std', 
                               'Best_Return', 'Worst_Return']
                st.dataframe(display_df[display_cols], use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
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
                    spy_total_return = (spy_benchmark.iloc[-1] - 1)
                    st.metric("SPY Buy & Hold Return", f"{spy_total_return:.2%}")
                
                # Visualization
                st.subheader("Performance by RSI Threshold")
                
                # Filter out thresholds with no trades for cleaner charts
                chart_data = results_df[results_df['Total_Trades'] > 0]
                
                if not chart_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.line(chart_data, x='RSI_Threshold', y='Total_Return', 
                                      title='Total Return by RSI Threshold',
                                      labels={'Total_Return': 'Total Return', 'RSI_Threshold': 'RSI Threshold'})
                        fig1.update_traces(line=dict(color='green'))
                        # Add SPY benchmark line
                        spy_return = (spy_benchmark.iloc[-1] - 1)
                        fig1.add_hline(y=spy_return, line_dash="dash", line_color="red", 
                                      annotation_text=f"SPY Buy & Hold: {spy_return:.2%}")
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
                    
                    # COMPARISON: BEST STRATEGIES
                    st.subheader("Strategy Comparison")
                    
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
                        
                        # Add SPY benchmark
                        fig_comparison.add_trace(go.Scatter(
                            x=spy_benchmark.index,
                            y=spy_benchmark.values,
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
                        st.subheader("Performance Summary")
                        
                        comparison_data = []
                        
                        # SPY data
                        spy_return = (spy_benchmark.iloc[-1] - 1)
                        comparison_data.append({
                            'Strategy': 'SPY Buy & Hold',
                            'Total Return': f"{spy_return:.2%}",
                            'Final Value': f"{spy_benchmark.iloc[-1]:.3f}",
                            'Sortino Ratio': 'N/A',
                            'Trades': 0
                        })
                        
                        # Best return strategy
                        best_return_data = results_df.loc[best_return_idx]
                        comparison_data.append({
                            'Strategy': f'Best Return (RSI {best_return_threshold})',
                            'Total Return': f"{best_return_data['Total_Return']:.2%}",
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
                                'Final Value': f"{best_sortino_data['Final_Equity']:.3f}",
                                'Sortino Ratio': f"{best_sortino_data['Sortino_Ratio']:.2f}",
                                'Trades': best_sortino_data['Total_Trades']
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                    
                    else:
                        st.warning("No valid Sortino ratios calculated (all strategies may have no downside deviation)")

                    st.subheader("Equity Curve - Best Cumulative Return Strategy")
                    best_idx = results_df['Total_Return'].idxmax()
                    best_threshold = results_df.loc[best_idx, 'RSI_Threshold']
                    best_equity_curve = results_df.loc[best_idx, 'equity_curve']
                    best_trades = results_df.loc[best_idx, 'trades']
                    best_total_return = results_df.loc[best_idx, 'Total_Return']
                    
                    st.info(f"📊 **Best Cumulative Return:** RSI {comparison.replace('_', ' ').title()} {best_threshold} | **Total Return:** {best_total_return:.2%}")
                    
                    if best_equity_curve is not None:
                        # Create equity curve chart
                        fig_equity = go.Figure()
                        
                        fig_equity.add_trace(go.Scatter(
                            x=best_equity_curve.index,
                            y=best_equity_curve.values,
                            mode='lines',
                            name='Equity Curve',
                            line=dict(color='darkgreen', width=2)
                        ))
                        
                        # Add trade markers
                        if best_trades:
                            entry_dates = [trade['entry_date'] for trade in best_trades]
                            entry_values = [best_equity_curve[trade['entry_date']] for trade in best_trades]
                            exit_dates = [trade['exit_date'] for trade in best_trades]
                            exit_values = [best_equity_curve[trade['exit_date']] for trade in best_trades]
                            
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
                            title=f"Equity Curve - RSI {comparison.replace('_', ' ').title()} {best_threshold} (Best Cumulative Return)",
                            xaxis_title="Date",
                            yaxis_title="Equity Value",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_equity, use_container_width=True)
                        
                        # Trade details for best cumulative return strategy
                        if best_trades:
                            with st.expander(f"Trade Details - Best Cumulative Return (RSI {best_threshold})"):
                                trades_df = pd.DataFrame(best_trades)
                                trades_df['return'] = trades_df['return'].apply(lambda x: f"{x:.2%}")
                                trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
                                trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
                                st.dataframe(trades_df, use_container_width=True)
                    
                    # USER-SELECTED EQUITY CURVE
                    st.subheader("View Any Strategy's Equity Curve")
                    
                    # Create dropdown with available thresholds that have trades
                    available_thresholds = results_df[results_df['Total_Trades'] > 0]['RSI_Threshold'].tolist()
                    
                    if available_thresholds:
                        selected_threshold = st.selectbox(
                            "Select RSI Threshold to view equity curve:",
                            options=available_thresholds,
                            index=available_thresholds.index(best_threshold) if best_threshold in available_thresholds else 0,
                            format_func=lambda x: f"RSI {comparison.replace('_', ' ').title()} {x}"
                        )
                        
                        # Get data for selected strategy
                        selected_idx = results_df[results_df['RSI_Threshold'] == selected_threshold].index[0]
                        selected_equity_curve = results_df.loc[selected_idx, 'equity_curve']
                        selected_trades = results_df.loc[selected_idx, 'trades']
                        selected_total_return = results_df.loc[selected_idx, 'Total_Return']
                        selected_win_rate = results_df.loc[selected_idx, 'Win_Rate']
                        selected_total_trades = results_df.loc[selected_idx, 'Total_Trades']
                        
                        # Show metrics for selected strategy
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Return", f"{selected_total_return:.2%}")
                        with col2:
                            st.metric("Win Rate", f"{selected_win_rate:.1%}")
                        with col3:
                            st.metric("Total Trades", selected_total_trades)
                        
                        # Create equity curve chart for selected strategy
                        if selected_equity_curve is not None:
                            fig_selected = go.Figure()
                            
                            fig_selected.add_trace(go.Scatter(
                                x=selected_equity_curve.index,
                                y=selected_equity_curve.values,
                                mode='lines',
                                name='Equity Curve',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add trade markers
                            if selected_trades:
                                entry_dates = [trade['entry_date'] for trade in selected_trades]
                                entry_values = [selected_equity_curve[trade['entry_date']] for trade in selected_trades]
                                exit_dates = [trade['exit_date'] for trade in selected_trades]
                                exit_values = [selected_equity_curve[trade['exit_date']] for trade in selected_trades]
                                
                                fig_selected.add_trace(go.Scatter(
                                    x=entry_dates,
                                    y=entry_values,
                                    mode='markers',
                                    name='Buy',
                                    marker=dict(color='green', size=8, symbol='triangle-up')
                                ))
                                
                                fig_selected.add_trace(go.Scatter(
                                    x=exit_dates,
                                    y=exit_values,
                                    mode='markers',
                                    name='Sell',
                                    marker=dict(color='red', size=8, symbol='triangle-down')
                                ))
                            
                            fig_selected.update_layout(
                                title=f"Equity Curve - RSI {comparison.replace('_', ' ').title()} {selected_threshold}",
                                xaxis_title="Date",
                                yaxis_title="Equity Value",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_selected, use_container_width=True)
                            
                            # Trade details for selected strategy
                            if selected_trades:
                                with st.expander(f"Trade Details - RSI {selected_threshold}"):
                                    trades_df = pd.DataFrame(selected_trades)
                                    trades_df['return'] = trades_df['return'].apply(lambda x: f"{x:.2%}")
                                    trades_df['entry_price'] = trades_df['entry_price'].apply(lambda x: f"${x:.2f}")
                                    trades_df['exit_price'] = trades_df['exit_price'].apply(lambda x: f"${x:.2f}")
                                    st.dataframe(trades_df, use_container_width=True)
                
                # Download results
                csv = results_df[display_cols].to_csv(index=False)
                filename_suffix = f"_{start_date}_{end_date}" if use_date_range and start_date and end_date else "_max_range"
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}{filename_suffix}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        if rsi_min >= rsi_max:
            st.error("Please ensure RSI Min is less than RSI Max")
        if use_date_range and (not start_date or not end_date or start_date >= end_date):
            st.error("Please ensure start date is before end date")

st.write("---")
st.write("💡 **Tip:** Try different ticker combinations and RSI conditions to find optimal signal thresholds")
