import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.title("RSI Backtest Engine")
st.write("Interactive RSI Trading Strategy Backtester")

def calculate_rsi_simple(prices, window=10):
    """Simple RSI calculation"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_test_data():
    """Generate simple test data"""
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    
    # Simple random walk
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    
    prices = [100]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices[1:], index=dates)

def simple_backtest():
    """Run a simple backtest"""
    with st.spinner("Generating test data..."):
        prices = generate_test_data()
    
    with st.spinner("Calculating RSI..."):
        rsi = calculate_rsi_simple(prices)
    
    # Simple strategy: buy when RSI < 30, sell when RSI > 70
    signals = pd.Series(0, index=prices.index)
    signals[rsi < 30] = 1   # Buy
    signals[rsi > 70] = -1  # Sell
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = signals.shift(1) * returns
    
    total_return = strategy_returns.sum()
    
    # Display results
    st.subheader("RSI Strategy Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Data Period", f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        st.metric("Trading Days", len(prices))
    
    with col2:
        st.metric("Final RSI", f"{rsi.iloc[-1]:.1f}")
        st.metric("Strategy Return", f"{total_return:.2%}")
    
    with col3:
        st.metric("Buy Signals", (signals == 1).sum())
        st.metric("Sell Signals", (signals == -1).sum())
    
    # Create charts
    st.subheader("Price Chart with RSI Signals")
    
    # Price chart with buy/sell signals
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices.values,
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add buy signals
    buy_signals = prices[signals == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals.values,
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    # Add sell signals
    sell_signals = prices[signals == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals.values,
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    fig.update_layout(title="Price Chart with Trading Signals", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)
    
    # RSI chart
    st.subheader("RSI Indicator")
    fig_rsi = go.Figure()
    
    fig_rsi.add_trace(go.Scatter(
        x=rsi.index,
        y=rsi.values,
        mode='lines',
        name='RSI',
        line=dict(color='purple')
    ))
    
    # Add RSI levels
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    fig_rsi.update_layout(
        title="RSI Indicator",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    
    return {
        'prices': prices,
        'rsi': rsi,
        'signals': signals,
        'total_return': total_return
    }

# Main app execution
st.write("Click the button below to run the RSI backtest:")

if st.button("Run Backtest", type="primary"):
    try:
        st.write("Starting backtest...")
        results = simple_backtest()
        st.success("✓ Backtest completed successfully!")
        
        # Additional analysis
        with st.expander("View Raw Data"):
            df = pd.DataFrame({
                'Price': results['prices'],
                'RSI': results['rsi'],
                'Signal': results['signals']
            })
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error: {e}")

st.write("---")
st.write("RSI Backtest Engine loaded and ready!")
