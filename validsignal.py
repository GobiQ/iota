import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(ticker, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for ticker {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def generate_signals(data, rsi_oversold, rsi_overbought, rsi_window):
    """Generate buy/sell signals based on RSI conditions"""
    data['RSI'] = calculate_rsi(data['Close'], window=rsi_window)
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'][data['RSI'] <= rsi_oversold] = 1  # Buy signal
    data['Signal'][data['RSI'] >= rsi_overbought] = -1  # Sell signal
    
    # Create position changes
    data['Position'] = data['Signal'].replace(to_replace=0, method='ffill')
    data['Position_Change'] = data['Position'].diff()
    
    return data

def backtest_strategy(signal_data, vix_data, hold_days=5):
    """Perform backtesting of the strategy"""
    trades = []
    current_position = None
    
    for i in range(len(signal_data)):
        if signal_data['Position_Change'].iloc[i] != 0:
            signal_type = signal_data['Position_Change'].iloc[i]
            entry_date = signal_data.index[i]
            entry_price = vix_data['Close'].loc[entry_date] if entry_date in vix_data.index else None
            
            if entry_price is None:
                continue
                
            # Calculate exit date
            exit_date = entry_date + pd.Timedelta(days=hold_days)
            
            # Find the closest trading day for exit
            available_dates = vix_data.index[vix_data.index >= exit_date]
            if len(available_dates) > 0:
                actual_exit_date = available_dates[0]
                exit_price = vix_data['Close'].loc[actual_exit_date]
                
                # Calculate return based on signal type
                if signal_type == 1:  # Buy signal (long VIX)
                    trade_return = (exit_price - entry_price) / entry_price
                else:  # Sell signal (short VIX)
                    trade_return = (entry_price - exit_price) / entry_price
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': actual_exit_date,
                    'Signal_Type': 'Long' if signal_type == 1 else 'Short',
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Return': trade_return,
                    'Hold_Days': (actual_exit_date - entry_date).days
                })
    
    return pd.DataFrame(trades)

def calculate_metrics(trades_df):
    """Calculate performance metrics"""
    if trades_df.empty:
        return {}
    
    returns = trades_df['Return']
    
    metrics = {
        'Total Trades': len(trades_df),
        'Win Rate': (returns > 0).mean() * 100,
        'Average Return': returns.mean() * 100,
        'Average Winner': returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
        'Average Loser': returns[returns < 0].mean() * 100 if (returns < 0).any() else 0,
        'Best Trade': returns.max() * 100,
        'Worst Trade': returns.min() * 100,
        'Profit Factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if (returns < 0).any() else float('inf'),
        'Sharpe Ratio': returns.mean() / returns.std() if returns.std() != 0 else 0,
        'Max Drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min() * 100
    }
    
    return metrics

# Streamlit App
st.set_page_config(page_title="RSI Signal Validation Backtest Engine", layout="wide")

st.title("🎯 RSI Signal Validation Backtest Engine")
st.markdown("Analyze trading signals based on RSI conditions with VIX/UVXY instruments")

# Sidebar for inputs
st.sidebar.header("📊 Strategy Parameters")

# Ticker selection
signal_ticker = st.sidebar.text_input("Signal Ticker (e.g., SPY)", value="SPY")

# VIX instrument selection
vix_instrument = st.sidebar.selectbox(
    "VIX Instrument",
    ["VIXY", "UVXY", "VXX", "SVXY", "XIV"],
    index=0
)

# RSI parameters
rsi_window = st.sidebar.slider("RSI Window", min_value=5, max_value=30, value=14)
rsi_oversold = st.sidebar.slider("RSI Oversold Level", min_value=10, max_value=40, value=30)
rsi_overbought = st.sidebar.slider("RSI Overbought Level", min_value=60, max_value=90, value=70)

# Date range
st.sidebar.subheader("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

# Hold period
hold_days = st.sidebar.slider("Hold Period (Days)", min_value=1, max_value=30, value=5)

# Run backtest button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

if run_backtest:
    with st.spinner("Fetching data and running backtest..."):
        # Fetch data
        signal_data = fetch_data(signal_ticker, start_date, end_date)
        vix_data = fetch_data(vix_instrument, start_date, end_date)
        
        if signal_data is not None and vix_data is not None:
            # Generate signals
            signal_data = generate_signals(signal_data, rsi_oversold, rsi_overbought, rsi_window)
            
            # Run backtest
            trades_df = backtest_strategy(signal_data, vix_data, hold_days)
            
            if not trades_df.empty:
                # Calculate metrics
                metrics = calculate_metrics(trades_df)
                
                # Display results
                st.header("📈 Backtest Results")
                
                # Metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", f"{metrics['Total Trades']}")
                    st.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
                    st.metric("Avg Return", f"{metrics['Average Return']:.2f}%")
                
                with col2:
                    st.metric("Best Trade", f"{metrics['Best Trade']:.2f}%")
                    st.metric("Worst Trade", f"{metrics['Worst Trade']:.2f}%")
                    st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
                
                with col3:
                    st.metric("Avg Winner", f"{metrics['Average Winner']:.2f}%")
                    st.metric("Avg Loser", f"{metrics['Average Loser']:.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                
                with col4:
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")
                    long_trades = len(trades_df[trades_df['Signal_Type'] == 'Long'])
                    short_trades = len(trades_df[trades_df['Signal_Type'] == 'Short'])
                    st.metric("Long Trades", f"{long_trades}")
                    st.metric("Short Trades", f"{short_trades}")
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Return distribution
                    fig_dist = px.histogram(
                        trades_df, 
                        x='Return',
                        nbins=20,
                        title="Distribution of Trade Returns",
                        labels={'Return': 'Return (%)', 'count': 'Frequency'}
                    )
                    fig_dist.update_xaxis(tickformat='.1%')
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Win/Loss by signal type
                    trades_df['Win_Loss'] = trades_df['Return'].apply(lambda x: 'Win' if x > 0 else 'Loss')
                    win_loss_summary = trades_df.groupby(['Signal_Type', 'Win_Loss']).size().reset_index(name='Count')
                    
                    fig_wl = px.bar(
                        win_loss_summary,
                        x='Signal_Type',
                        y='Count',
                        color='Win_Loss',
                        title="Wins vs Losses by Signal Type",
                        color_discrete_map={'Win': 'green', 'Loss': 'red'}
                    )
                    st.plotly_chart(fig_wl, use_container_width=True)
                
                # Cumulative returns chart
                trades_df['Cumulative_Return'] = (1 + trades_df['Return']).cumprod() - 1
                
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=trades_df['Exit_Date'],
                    y=trades_df['Cumulative_Return'] * 100,
                    mode='lines+markers',
                    name='Cumulative Return',
                    line=dict(color='blue', width=2)
                ))
                
                fig_cum.update_layout(
                    title="Cumulative Returns Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_cum, use_container_width=True)
                
                # Signal chart
                st.subheader(f"RSI Signals for {signal_ticker}")
                
                # Create subplot with RSI and price
                fig_signals = make_subplots(
                    rows=2, cols=1,
                    shared_xaxis=True,
                    vertical_spacing=0.1,
                    subplot_titles=[f'{signal_ticker} Price', 'RSI'],
                    row_heights=[0.7, 0.3]
                )
                
                # Price chart
                fig_signals.add_trace(
                    go.Scatter(x=signal_data.index, y=signal_data['Close'], 
                              name='Price', line=dict(color='black')),
                    row=1, col=1
                )
                
                # Buy signals
                buy_signals = signal_data[signal_data['Position_Change'] == 1]
                if not buy_signals.empty:
                    fig_signals.add_trace(
                        go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                  mode='markers', name='Buy Signal',
                                  marker=dict(color='green', size=10, symbol='triangle-up')),
                        row=1, col=1
                    )
                
                # Sell signals
                sell_signals = signal_data[signal_data['Position_Change'] == -1]
                if not sell_signals.empty:
                    fig_signals.add_trace(
                        go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                  mode='markers', name='Sell Signal',
                                  marker=dict(color='red', size=10, symbol='triangle-down')),
                        row=1, col=1
                    )
                
                # RSI chart
                fig_signals.add_trace(
                    go.Scatter(x=signal_data.index, y=signal_data['RSI'],
                              name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                
                # RSI levels
                fig_signals.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
                fig_signals.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
                fig_signals.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
                
                fig_signals.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig_signals, use_container_width=True)
                
                # Trade details table
                st.subheader("📋 Individual Trades")
                trades_display = trades_df.copy()
                trades_display['Return'] = trades_display['Return'] * 100
                trades_display['Entry_Price'] = trades_display['Entry_Price'].round(2)
                trades_display['Exit_Price'] = trades_display['Exit_Price'].round(2)
                trades_display['Return'] = trades_display['Return'].round(2)
                
                st.dataframe(
                    trades_display[['Entry_Date', 'Exit_Date', 'Signal_Type', 'Entry_Price', 
                                   'Exit_Price', 'Return', 'Hold_Days']],
                    use_container_width=True
                )
                
            else:
                st.warning("No trades generated with the current parameters. Try adjusting RSI levels or date range.")
        else:
            st.error("Failed to fetch data. Please check ticker symbols and try again.")

else:
    st.info("👈 Configure your parameters in the sidebar and click 'Run Backtest' to start the analysis.")
    
    # Show example configuration
    st.subheader("💡 Example Strategy")
    st.markdown("""
    **Sample Configuration:**
    - **Signal Ticker**: SPY (tracks overall market)
    - **VIX Instrument**: VIXY (VIX ETF)
    - **RSI Window**: 14 days
    - **Oversold Level**: 30 (buy VIX when market oversold)
    - **Overbought Level**: 70 (sell/short VIX when market overbought)
    - **Hold Period**: 5 days
    
    This strategy assumes that when the market (SPY) becomes oversold (low RSI), 
    volatility (VIX instruments) will increase, and vice versa.
    """)

# Footer
st.markdown("---")
st.markdown("**Note**: This tool is for educational and research purposes only. Past performance does not guarantee future results.")
