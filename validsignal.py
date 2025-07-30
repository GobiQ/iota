import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

st.title("RSI Signal Analysis Engine")
st.write("Analyze RSI signals across different thresholds for any ticker pair")

def calculate_rsi(prices, window=14):
    """Calculate RSI with standard 14-period window"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_stock_data(ticker, period="2y"):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            st.error(f"No data found for ticker: {ticker}")
            return None
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def analyze_rsi_signals(signal_prices, target_prices, rsi_threshold, signal_type="buy"):
    """Analyze RSI signals for a specific threshold"""
    rsi = calculate_rsi(signal_prices)
    
    # Generate signals based on RSI threshold
    if signal_type == "buy":
        signals = (rsi <= rsi_threshold).astype(int)
    else:  # sell
        signals = (rsi >= rsi_threshold).astype(int)
    
    # Find signal points (when signal changes from 0 to 1)
    signal_changes = signals.diff()
    entry_points = signal_changes == 1
    
    if not entry_points.any():
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'returns': []
        }
    
    # Calculate returns for each signal
    returns = []
    hold_period = 5  # Hold for 5 days after signal
    
    for entry_date in signal_prices[entry_points].index:
        try:
            entry_idx = target_prices.index.get_loc(entry_date)
            if entry_idx + hold_period < len(target_prices):
                entry_price = target_prices.iloc[entry_idx]
                exit_price = target_prices.iloc[entry_idx + hold_period]
                ret = (exit_price - entry_price) / entry_price
                returns.append(ret)
        except (KeyError, IndexError):
            continue
    
    if not returns:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'returns': []
        }
    
    returns = np.array(returns)
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()
    
    return {
        'total_trades': len(returns),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'returns': returns
    }

def run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, signal_type):
    """Run comprehensive RSI analysis across the specified range"""
    
    # Fetch data
    with st.spinner(f"Fetching data for {signal_ticker}..."):
        signal_data = get_stock_data(signal_ticker)
    
    with st.spinner(f"Fetching data for {target_ticker}..."):
        target_data = get_stock_data(target_ticker)
    
    if signal_data is None or target_data is None:
        return None
    
    # Align data on common dates
    common_dates = signal_data.index.intersection(target_data.index)
    signal_data = signal_data[common_dates]
    target_data = target_data[common_dates]
    
    # Generate RSI thresholds (every 0.5)
    rsi_thresholds = np.arange(rsi_min, rsi_max + 0.5, 0.5)
    
    results = []
    
    progress_bar = st.progress(0)
    total_thresholds = len(rsi_thresholds)
    
    for i, threshold in enumerate(rsi_thresholds):
        analysis = analyze_rsi_signals(signal_data, target_data, threshold, signal_type)
        
        results.append({
            'RSI_Threshold': threshold,
            'Total_Trades': analysis['total_trades'],
            'Win_Rate': analysis['win_rate'],
            'Avg_Return': analysis['avg_return'],
            'Return_Std': np.std(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Best_Return': np.max(analysis['returns']) if len(analysis['returns']) > 0 else 0,
            'Worst_Return': np.min(analysis['returns']) if len(analysis['returns']) > 0 else 0
        })
        
        progress_bar.progress((i + 1) / total_thresholds)
    
    return pd.DataFrame(results)

# Streamlit Interface
st.sidebar.header("Configuration")

# Input fields
signal_ticker = st.sidebar.text_input("Signal Ticker", value="SPY", help="Ticker to generate RSI signals from")
target_ticker = st.sidebar.text_input("Target Ticker", value="QQQ", help="Ticker to trade based on signals")

signal_type = st.sidebar.selectbox("Signal Type", ["buy", "sell"], help="Buy signals trigger on low RSI, sell signals on high RSI")

if signal_type == "buy":
    default_min, default_max = 20, 40
    st.sidebar.write("Buy signals: RSI <= threshold")
else:
    default_min, default_max = 60, 80
    st.sidebar.write("Sell signals: RSI >= threshold")

rsi_min = st.sidebar.number_input("RSI Range Min", min_value=0.0, max_value=100.0, value=float(default_min), step=0.5)
rsi_max = st.sidebar.number_input("RSI Range Max", min_value=0.0, max_value=100.0, value=float(default_max), step=0.5)

if rsi_min >= rsi_max:
    st.sidebar.error("RSI Min must be less than RSI Max")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Analysis Configuration")
    st.write(f"**Signal Ticker:** {signal_ticker}")
    st.write(f"**Target Ticker:** {target_ticker}")
    st.write(f"**Signal Type:** {signal_type.title()}")
    st.write(f"**RSI Range:** {rsi_min} - {rsi_max}")

with col2:
    st.subheader("Strategy Logic")
    if signal_type == "buy":
        st.info("🔵 Generate BUY signals when RSI ≤ threshold\n\n📈 Trade the target ticker for 5 days")
    else:
        st.info("🔴 Generate SELL signals when RSI ≥ threshold\n\n📉 Trade the target ticker for 5 days")

if st.button("Run RSI Analysis", type="primary"):
    if rsi_min < rsi_max:
        try:
            results_df = run_rsi_analysis(signal_ticker, target_ticker, rsi_min, rsi_max, signal_type)
            
            if results_df is not None and not results_df.empty:
                st.success("✓ Analysis completed successfully!")
                
                # Display results table
                st.subheader("RSI Analysis Results")
                
                # Format the dataframe for display
                display_df = results_df.copy()
                display_df['Win_Rate'] = display_df['Win_Rate'].apply(lambda x: f"{x:.1%}")
                display_df['Avg_Return'] = display_df['Avg_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Return_Std'] = display_df['Return_Std'].apply(lambda x: f"{x:.2%}")
                display_df['Best_Return'] = display_df['Best_Return'].apply(lambda x: f"{x:.2%}")
                display_df['Worst_Return'] = display_df['Worst_Return'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = results_df['Total_Trades'].sum()
                    st.metric("Total Signals Generated", total_trades)
                
                with col2:
                    avg_win_rate = results_df[results_df['Total_Trades'] > 0]['Win_Rate'].mean()
                    st.metric("Average Win Rate", f"{avg_win_rate:.1%}" if not np.isnan(avg_win_rate) else "N/A")
                
                with col3:
                    avg_return = results_df[results_df['Total_Trades'] > 0]['Avg_Return'].mean()
                    st.metric("Average Return", f"{avg_return:.2%}" if not np.isnan(avg_return) else "N/A")
                
                with col4:
                    best_threshold = results_df.loc[results_df['Avg_Return'].idxmax(), 'RSI_Threshold'] if not results_df.empty else "N/A"
                    st.metric("Best RSI Threshold", best_threshold)
                
                # Visualization
                st.subheader("Performance by RSI Threshold")
                
                # Filter out thresholds with no trades for cleaner charts
                chart_data = results_df[results_df['Total_Trades'] > 0]
                
                if not chart_data.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.line(chart_data, x='RSI_Threshold', y='Win_Rate', 
                                      title='Win Rate by RSI Threshold',
                                      labels={'Win_Rate': 'Win Rate', 'RSI_Threshold': 'RSI Threshold'})
                        fig1.update_traces(line=dict(color='green'))
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.line(chart_data, x='RSI_Threshold', y='Avg_Return',
                                      title='Average Return by RSI Threshold',
                                      labels={'Avg_Return': 'Average Return', 'RSI_Threshold': 'RSI Threshold'})
                        fig2.update_traces(line=dict(color='blue'))
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Trade frequency chart
                    fig3 = px.bar(chart_data, x='RSI_Threshold', y='Total_Trades',
                                 title='Number of Trades by RSI Threshold',
                                 labels={'Total_Trades': 'Number of Trades', 'RSI_Threshold': 'RSI Threshold'})
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"rsi_analysis_{signal_ticker}_{target_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.error("Please ensure RSI Min is less than RSI Max")

st.write("---")
st.write("💡 **Tip:** Try different ticker combinations and RSI ranges to find optimal signal thresholds")
