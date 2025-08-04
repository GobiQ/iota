import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Signal Discovery Trading App",
    page_icon="📈",
    layout="wide"
)

class SignalDiscovery:
    def __init__(self):
        self.data = {}
        self.signals = {}
        self.backtest_results = {}
        
    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical data for given tickers"""
        data = {}
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                if not df.empty:
                    data[ticker] = df
                    st.success(f"✅ Fetched data for {ticker}")
                else:
                    st.error(f"❌ No data found for {ticker}")
            except Exception as e:
                st.error(f"❌ Error fetching {ticker}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(tickers))
        
        return data
    
    def calculate_rsi(self, prices, period=10):
        """Calculate RSI for given prices"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_moving_averages(self, prices):
        """Calculate multiple moving averages"""
        ma_periods = [10, 20, 30, 100, 200, 300]
        mas = {}
        for period in ma_periods:
            mas[f'MA_{period}'] = prices.rolling(window=period).mean()
        return mas
    
    def generate_signals(self, data, signal_config):
        """Generate trading signals based on configuration"""
        all_signals = {}
        
        for ticker, df in data.items():
            signals = pd.DataFrame(index=df.index)
            
            # Calculate RSI
            rsi_10 = self.calculate_rsi(df['Close'], 10)
            signals['RSI_10'] = rsi_10
            
            # Calculate Moving Averages
            mas = self.calculate_moving_averages(df['Close'])
            for ma_name, ma_values in mas.items():
                signals[ma_name] = ma_values
                signals[f'{ma_name}_Signal'] = df['Close'] > ma_values
            
            signals['Close'] = df['Close']
            all_signals[ticker] = signals
        
        # Generate comparison signals
        for config in signal_config:
            if config['type'] == 'rsi_comparison':
                ticker1, ticker2 = config['ticker1'], config['ticker2']
                if ticker1 in all_signals and ticker2 in all_signals:
                    signal_name = f'RSI_{ticker1}_vs_{ticker2}'
                    all_signals[signal_name] = (
                        all_signals[ticker1]['RSI_10'] > all_signals[ticker2]['RSI_10']
                    )
            
            elif config['type'] == 'rsi_static':
                ticker = config['ticker']
                threshold = config['threshold']
                if ticker in all_signals:
                    signal_name = f'RSI_{ticker}_vs_{threshold}'
                    all_signals[signal_name] = all_signals[ticker]['RSI_10'] > threshold
        
        return all_signals
    
    def split_data_non_continuous(self, data, train_size=0.8, random_state=42):
        """Split data into non-continuous train/test sets"""
        dates = data.index.tolist()
        
        # Randomly split dates
        train_dates, test_dates = train_test_split(
            dates, train_size=train_size, random_state=random_state, shuffle=True
        )
        
        train_data = data.loc[train_dates].sort_index()
        test_data = data.loc[test_dates].sort_index()
        
        return train_data, test_data
    
    def backtest_strategy(self, signals, allocations, initial_capital=100000):
        """Backtest the trading strategy"""
        results = {}
        
        for ticker, allocation_pct in allocations.items():
            if ticker not in signals:
                continue
                
            df = signals[ticker].copy()
            df = df.dropna()
            
            if df.empty:
                continue
            
            # Split into train/test
            train_data, test_data = self.split_data_non_continuous(df)
            
            # Backtest on test data
            capital = initial_capital * (allocation_pct / 100)
            position = 0
            cash = capital
            portfolio_value = []
            trades = []
            
            for date, row in test_data.iterrows():
                current_price = row['Close']
                
                # Generate trading signal (example: MA crossover)
                signal = 0
                if pd.notna(row.get('MA_20')) and pd.notna(row.get('MA_50')):
                    if row.get('MA_20', 0) > row.get('MA_50', 0):
                        signal = 1  # Buy signal
                    else:
                        signal = -1  # Sell signal
                
                # Execute trades
                if signal == 1 and position <= 0:  # Buy
                    shares_to_buy = cash // current_price
                    if shares_to_buy > 0:
                        position += shares_to_buy
                        cash -= shares_to_buy * current_price
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price
                        })
                
                elif signal == -1 and position > 0:  # Sell
                    cash += position * current_price
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price
                    })
                    position = 0
                
                # Calculate portfolio value
                portfolio_val = cash + (position * current_price)
                portfolio_value.append({
                    'date': date,
                    'portfolio_value': portfolio_val,
                    'cash': cash,
                    'position': position,
                    'price': current_price
                })
            
            results[ticker] = {
                'portfolio_history': pd.DataFrame(portfolio_value),
                'trades': pd.DataFrame(trades),
                'train_period': f"{train_data.index.min()} to {train_data.index.max()}",
                'test_period': f"{test_data.index.min()} to {test_data.index.max()}",
                'final_value': portfolio_value[-1]['portfolio_value'] if portfolio_value else capital,
                'return_pct': ((portfolio_value[-1]['portfolio_value'] / capital - 1) * 100) if portfolio_value else 0
            }
        
        return results

# Streamlit App
def main():
    st.title("📈 Signal Discovery Trading Application")
    st.markdown("---")
    
    # Initialize session state
    if 'signal_discovery' not in st.session_state:
        st.session_state.signal_discovery = SignalDiscovery()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Ticker input
    st.sidebar.subheader("📊 Tickers")
    ticker_input = st.sidebar.text_area(
        "Enter tickers (one per line):",
        value="AAPL\nMSFT\nGOOGL\nTSLA",
        height=100
    )
    tickers = [t.strip().upper() for t in ticker_input.split('\n') if t.strip()]
    
    # Signal Configuration
    st.sidebar.subheader("🎯 Signal Configuration")
    
    signal_configs = []
    
    # RSI Comparison Signals
    st.sidebar.write("**RSI Comparisons:**")
    num_rsi_comp = st.sidebar.number_input("Number of RSI comparisons", 0, 10, 2)
    
    for i in range(num_rsi_comp):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker1 = st.selectbox(f"Ticker 1 (RSI {i+1})", tickers, key=f"rsi_t1_{i}")
        with col2:
            ticker2 = st.selectbox(f"Ticker 2 (RSI {i+1})", tickers, key=f"rsi_t2_{i}")
        
        signal_configs.append({
            'type': 'rsi_comparison',
            'ticker1': ticker1,
            'ticker2': ticker2
        })
    
    # RSI vs Static Value
    st.sidebar.write("**RSI vs Static Value:**")
    num_rsi_static = st.sidebar.number_input("Number of RSI static comparisons", 0, 10, 1)
    
    for i in range(num_rsi_static):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            ticker = st.selectbox(f"Ticker (Static {i+1})", tickers, key=f"rsi_static_t_{i}")
        with col2:
            threshold = st.number_input(f"RSI Threshold {i+1}", 0, 100, 50, key=f"rsi_thresh_{i}")
        
        signal_configs.append({
            'type': 'rsi_static',
            'ticker': ticker,
            'threshold': threshold
        })
    
    # Allocation Configuration
    st.sidebar.subheader("💰 Allocations")
    allocations = {}
    remaining_allocation = 100.0
    
    for ticker in tickers:
        max_allocation = min(remaining_allocation, 100.0)
        allocation = st.sidebar.slider(
            f"{ticker} allocation %",
            0.0, max_allocation, 
            min(25.0, max_allocation),
            key=f"alloc_{ticker}"
        )
        allocations[ticker] = allocation
        remaining_allocation -= allocation
    
    st.sidebar.write(f"Remaining allocation: {remaining_allocation:.1f}%")
    
    # Main content
    if st.button("🚀 Run Analysis", type="primary"):
        if not tickers:
            st.error("Please enter at least one ticker symbol")
            return
        
        with st.spinner("Fetching data and running analysis..."):
            # Fetch data
            data = st.session_state.signal_discovery.fetch_data(tickers, start_date, end_date)
            
            if not data:
                st.error("No data fetched. Please check your ticker symbols.")
                return
            
            # Generate signals
            signals = st.session_state.signal_discovery.generate_signals(data, signal_configs)
            
            # Run backtest
            backtest_results = st.session_state.signal_discovery.backtest_strategy(
                signals, allocations
            )
            
            # Display results
            st.markdown("## 📊 Analysis Results")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Signals", "Trades", "Detailed Analysis"])
            
            with tab1:
                st.subheader("Portfolio Performance")
                
                # Overall performance metrics
                total_return = 0
                total_trades = 0
                
                performance_data = []
                for ticker, results in backtest_results.items():
                    performance_data.append({
                        'Ticker': ticker,
                        'Final Value': f"${results['final_value']:,.2f}",
                        'Return %': f"{results['return_pct']:.2f}%",
                        'Trades': len(results['trades']),
                        'Train Period': results['train_period'],
                        'Test Period': results['test_period']
                    })
                    total_return += results['return_pct'] * (allocations.get(ticker, 0) / 100)
                    total_trades += len(results['trades'])
                
                # Display performance table
                if performance_data:
                    df_performance = pd.DataFrame(performance_data)
                    st.dataframe(df_performance, use_container_width=True)
                    
                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Weighted Return", f"{total_return:.2f}%")
                    with col2:
                        st.metric("Total Trades", total_trades)
                    with col3:
                        st.metric("Active Tickers", len(backtest_results))
                
                # Plot portfolio performance
                for ticker, results in backtest_results.items():
                    if not results['portfolio_history'].empty:
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=results['portfolio_history']['date'],
                            y=results['portfolio_history']['portfolio_value'],
                            mode='lines',
                            name=f'{ticker} Portfolio Value',
                            line=dict(width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{ticker} Portfolio Performance",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Signal Analysis")
                
                # Display RSI and MA signals for each ticker
                for ticker in tickers:
                    if ticker in signals:
                        st.write(f"**{ticker} Signals**")
                        
                        # Create signal plot
                        fig = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=[f'{ticker} Price & Moving Averages', f'{ticker} RSI'],
                            vertical_spacing=0.1
                        )
                        
                        signal_data = signals[ticker].dropna()
                        
                        # Price and MAs
                        fig.add_trace(
                            go.Scatter(x=signal_data.index, y=signal_data['Close'], 
                                     name='Close Price', line=dict(color='black', width=2)),
                            row=1, col=1
                        )
                        
                        ma_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
                        for i, ma_period in enumerate([10, 20, 30, 100, 200, 300]):
                            ma_col = f'MA_{ma_period}'
                            if ma_col in signal_data.columns:
                                fig.add_trace(
                                    go.Scatter(x=signal_data.index, y=signal_data[ma_col],
                                             name=f'MA {ma_period}', 
                                             line=dict(color=ma_colors[i % len(ma_colors)])),
                                    row=1, col=1
                                )
                        
                        # RSI
                        fig.add_trace(
                            go.Scatter(x=signal_data.index, y=signal_data['RSI_10'],
                                     name='RSI (10)', line=dict(color='purple')),
                            row=2, col=1
                        )
                        
                        # RSI reference lines
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                        
                        fig.update_layout(height=600, showlegend=True)
                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="RSI", row=2, col=1)
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Trade History")
                
                for ticker, results in backtest_results.items():
                    if not results['trades'].empty:
                        st.write(f"**{ticker} Trades**")
                        st.dataframe(results['trades'], use_container_width=True)
                        st.write("---")
            
            with tab4:
                st.subheader("Detailed Analysis")
                
                # Train/Test split visualization
                st.write("**Data Split Information**")
                for ticker, results in backtest_results.items():
                    st.write(f"**{ticker}:**")
                    st.write(f"- Training Period: {results['train_period']}")
                    st.write(f"- Testing Period: {results['test_period']}")
                    st.write(f"- Final Return: {results['return_pct']:.2f}%")
                    st.write("---")
                
                # Signal configuration summary
                st.write("**Signal Configuration:**")
                for i, config in enumerate(signal_configs):
                    if config['type'] == 'rsi_comparison':
                        st.write(f"{i+1}. RSI Comparison: {config['ticker1']} vs {config['ticker2']}")
                    elif config['type'] == 'rsi_static':
                        st.write(f"{i+1}. RSI Static: {config['ticker']} vs {config['threshold']}")

if __name__ == "__main__":
    main()