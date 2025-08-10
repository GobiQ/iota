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
        ticker_pairs = list(itertools.product(tickers, repeat=2))
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

def backtest_signals(signals: dict, price_data: pd.DataFrame, tickers: list, target_tickers: list):
    """Backtest individual signals"""
    log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
    results = []

    total_signals = len(signals) * len(target_tickers)
    progress = SimpleProgress(total_signals)

    for target_ticker in target_tickers:
        cumulative_returns = np.exp(log_returns[target_ticker].cumsum()) - 1

        for signal_name, signal in signals.items():
            progress.update(1)

            shifted_signal = signal.reindex(price_data.index).fillna(False).shift(1).fillna(False)
            returns = shifted_signal * log_returns[target_ticker]
            cumulative_returns_signal = np.exp(returns.cumsum()) - 1

            downside_returns = returns[returns < 0]
            sortino_ratio = returns.mean() / (downside_returns.std() + 1e-9) * np.sqrt(252)

            running_max = cumulative_returns_signal.cummax()
            drawdown = cumulative_returns_signal - running_max
            max_drawdown = drawdown.min()

            total_return = cumulative_returns_signal.iloc[-1]
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
            time_in_market = shifted_signal.mean()

            gross_profit = returns[returns > 0].sum()
            gross_loss = -returns[returns < 0].sum()
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

            active_returns = returns[shifted_signal]
            percent_profitable = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan

            results.append({
                'Signal': signal_name,
                'Ticker': target_ticker,
                'Total Return': total_return,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Max Drawdown': max_drawdown,
                'Time in Market': time_in_market,
                'Profit Factor': profit_factor,
                'Percent Profitable': percent_profitable,
                'Signal Returns': returns
            })

    progress.close()
    return pd.DataFrame(results).sort_values(by='Sortino Ratio', ascending=False)

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
    results = []
    
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

        results.append({
            'Signal': '+'.join(signal_names),
            'Ticker': ticker,
            'Total Return': total_return,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Time in Market': time_in_market,
            'Profit Factor': profit_factor,
            'Percent Profitable': percent_profitable,
            'Signal Returns': returns
        })

    progress.close()
    return pd.DataFrame(results).sort_values(by='Sortino Ratio', ascending=False)

def plot_performance_chart(results_df, selected_signals, price_data):
    """Create interactive performance chart"""
    fig = go.Figure()
    
    log_returns = np.log(price_data / price_data.shift(1)).fillna(0)
    
    for signal_name in selected_signals:
        signal_data = results_df[results_df['Signal'] == signal_name].iloc[0]
        returns = signal_data['Signal Returns']
        cumulative_returns = np.exp(returns.cumsum()) - 1
        
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name=f"{signal_name} ({signal_data['Ticker']})",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Cumulative Returns Comparison",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        height=500,
        showlegend=True
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
            
            # Backtest individual signals
            st.header("📈 Backtesting Individual Signals...")
            backtest_results = backtest_signals(signals, price_data, all_tickers, target)
            backtest_results = backtest_results.dropna()
            
            # Apply filters
            backtest_results = backtest_results[backtest_results['Time in Market'] > tim]
            backtest_results = backtest_results[backtest_results['Max Drawdown'] > mdd]
            backtest_results = backtest_results[backtest_results['Total Return'] > 0]
            
            st.info(f"📊 {len(backtest_results)} signals passed initial filters")
            
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
                    backtest_df_combined = backtest_combined_signals(combinations_filtered, signals, price_data, log_returns)
                    
                    # Combine results
                    backtest_df_combined = backtest_df_combined[backtest_results.columns]
                    final_results = pd.concat([backtest_filtered, backtest_df_combined])
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
                st.metric("Best Sortino Ratio", f"{final_results['Sortino Ratio'].max():.2f}")
            with col3:
                st.metric("Best Total Return", f"{final_results['Total Return'].max():.1%}")
            with col4:
                st.metric("Avg Time in Market", f"{final_results['Time in Market'].mean():.1%}")
            
            # Results table
            st.subheader("📊 Top Performing Signals")
            display_columns = ['Signal', 'Ticker', 'Total Return', 'Sortino Ratio', 'Calmar Ratio', 
                             'Max Drawdown', 'Time in Market', 'Profit Factor', 'Percent Profitable']
            
            # Format the dataframe for display
            display_df = final_results[display_columns].copy()
            display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.1%}")
            display_df['Sortino Ratio'] = display_df['Sortino Ratio'].apply(lambda x: f"{x:.2f}")
            display_df['Calmar Ratio'] = display_df['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.1%}")
            display_df['Time in Market'] = display_df['Time in Market'].apply(lambda x: f"{x:.1%}")
            display_df['Profit Factor'] = display_df['Profit Factor'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
            display_df['Percent Profitable'] = display_df['Percent Profitable'].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
            
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
                    chart = plot_performance_chart(final_results, selected_signals, price_data)
                    st.plotly_chart(chart, use_container_width=True)
            
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
            
        except Exception as e:
            st.error(f"An error occurred during backtesting: {str(e)}")
            st.exception(e)
    
    # Show cached results if available
    elif 'final_results' in st.session_state:
        st.info("Showing previous results. Click 'Run Backtest' to generate new results.")
        final_results = st.session_state['final_results']
        price_data = st.session_state['price_data']
        
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
