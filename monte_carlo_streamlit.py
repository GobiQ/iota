import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta, date
import os
import requests
from typing import List, Dict
import tempfile
import zipfile
import io

# Set page config
st.set_page_config(
    page_title="Monte Carlo Portfolio Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed for reproducibility
np.random.seed(42)

def convert_trading_date(date_int):
    """Convert trading date integer to datetime object"""
    date_1 = datetime.strptime("01/01/1970", "%m/%d/%Y")
    dt = date_1 + timedelta(days=int(date_int))
    return dt

class YahooFinanceAPI:
    """Fetches historical price data using the yfinance package."""
    
    def __init__(self, session=None):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            st.error("yfinance package is not installed. Please install it with: pip install yfinance")
            st.stop()
        
        self.ticker_map = {'BRK/B': 'BRK-B'}
        self.rate_limit_delay = 1.0
        self.use_batch_download = True
        self.batch_size = 5

    def fetch_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Fetch historical price data for multiple symbols."""
        
        # Map symbols to Yahoo Finance format if needed
        mapped_symbols = {}
        for symbol in symbols:
            yahoo_symbol = self.ticker_map.get(symbol, symbol)
            mapped_symbols[yahoo_symbol] = symbol

        return self._individual_download(mapped_symbols, start_date, end_date)

    def _individual_download(self, mapped_symbols: Dict[str, str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """Download data for each symbol individually."""
        import time
        
        price_data = {}
        progress_bar = st.progress(0)
        total_symbols = len(mapped_symbols)
        
        for i, (yahoo_symbol, original_symbol) in enumerate(mapped_symbols.items()):
            progress_bar.progress((i + 1) / total_symbols)
            
            try:
                ticker_obj = self.yf.Ticker(yahoo_symbol)
                data = ticker_obj.history(
                    start=start_date, 
                    end=end_date,
                    auto_adjust=True
                )

                if data.empty:
                    continue

                if 'Close' in data.columns:
                    series = data['Close'].copy()
                    series = series.dropna()
                    series = series.astype(np.float32)

                    if series.index.tz is not None:
                        series.index = series.index.tz_convert('America/New_York')
                        series.index = series.index.tz_localize(None)

                    series = series[~series.index.duplicated(keep='last')]

                    if not series.empty:
                        series.name = original_symbol
                        price_data[original_symbol] = series

            except Exception as e:
                st.warning(f"Error fetching data for {original_symbol}: {str(e)}")

            time.sleep(self.rate_limit_delay)

        progress_bar.empty()
        return price_data

@st.cache_data
def fetch_backtest(id, start_date, end_date):
    """Fetch backtest data from Composer API"""
    if id.endswith('/details'):
        id = id.split('/')[-2]
    else:
        id = id.split('/')[-1]

    payload = {
        "capital": 100000,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": 0.0005,
        "start_date": start_date,
        "end_date": end_date,
    }

    url = f"https://backtest-api.composer.trade/api/v2/public/symphonies/{id}/backtest"

    try:
        data = requests.post(url, json=payload)
        jsond = data.json()
        symphony_name = jsond['legend'][id]['name']

        holdings = jsond["last_market_days_holdings"]
        tickers = list(holdings.keys())

        allocations = jsond["tdvm_weights"]
        date_range = pd.date_range(start=start_date, end=end_date)
        df = pd.DataFrame(0.0, index=date_range, columns=tickers)

        for ticker in allocations:
            for date_int in allocations[ticker]:
                trading_date = convert_trading_date(date_int)
                percent = allocations[ticker][date_int]
                df.at[trading_date, ticker] = percent

        return df, symphony_name, tickers
    except Exception as e:
        st.error(f"Error fetching data from Composer: {str(e)}")
        return None, None, None

@st.cache_data
def calculate_portfolio_returns(allocations_df, tickers):
    """Calculate daily portfolio returns with proper allocation weighting"""
    
    # Find the first row with at least one non-zero value
    first_valid_index = allocations_df[(abs(allocations_df) > 0.000001).any(axis=1)].first_valid_index()
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0

    if '$USD' not in allocations_df.columns:
        allocations_df['$USD'] = 0

    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()
    
    unique_tickers = {ticker for ticker in tickers if ticker != '$USD'}
    
    start_date = allocations_df.index.min() - timedelta(days=10)
    end_date = allocations_df.index.max() + timedelta(days=10)

    yahoo_api = YahooFinanceAPI()
    prices_data = yahoo_api.fetch_historical_data(
        list(unique_tickers),
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    prices = pd.DataFrame({ticker: prices_data[ticker] for ticker in prices_data})
    prices.index = pd.to_datetime(prices.index).normalize()
    prices['$USD'] = 1.0

    for ticker in tickers:
        if ticker not in prices.columns and ticker != '$USD':
            prices[ticker] = np.nan

    prices = prices.ffill().bfill().fillna(1.0)
    prices = prices[tickers]

    allocations_df.sort_index(inplace=True)
    prices.sort_index(inplace=True)

    # Calculate price changes
    price_changes = {}
    price_dates = sorted(prices.index)
    
    for ticker in tickers:
        if ticker == '$USD':
            continue
        
        price_changes[ticker] = {}
        ticker_prices = prices[ticker]
        
        for i in range(1, len(price_dates)):
            today = price_dates[i]
            yesterday = price_dates[i - 1]
            
            today_price = ticker_prices.loc[today]
            yesterday_price = ticker_prices.loc[yesterday]
            
            if yesterday_price is not None:
                daily_change = ((today_price / yesterday_price) - 1) * 100
                price_changes[ticker][today.strftime('%Y-%m-%d')] = daily_change

    # Calculate portfolio returns
    daily_returns = pd.Series(index=allocations_df.index[1:], dtype=float)

    for i in range(1, len(allocations_df)):
        today_date = allocations_df.index[i]
        today_key = today_date.strftime('%Y-%m-%d')
        
        allocations_yday = allocations_df.iloc[i - 1, :] / 100.0
        
        portfolio_daily_return = 0.0
        
        for ticker in tickers:
            if ticker == '$USD':
                continue
                
            ticker_allocation = allocations_yday[ticker]
            
            if ticker_allocation > 0:
                if today_key in price_changes.get(ticker, {}):
                    ticker_return = price_changes[ticker][today_key]
                    portfolio_daily_return += ticker_allocation * ticker_return

        daily_returns.iloc[i - 1] = portfolio_daily_return

    return daily_returns, allocations_df.index

def run_monte_carlo_simulation(returns, num_simulations=10000, simulation_length=None, annual_periods=252):
    """Run Monte Carlo simulation using separate sampling for positive and negative returns"""
    if simulation_length is None:
        simulation_length = len(returns)

    returns_array = np.array(returns)
    
    positive_returns = returns_array[returns_array > 0]
    negative_returns = returns_array[returns_array <= 0]
    
    prob_positive = len(positive_returns) / len(returns_array)

    cumulative_returns = np.zeros((num_simulations, simulation_length + 1))
    cumulative_returns[:, 0] = 0

    sharpe_ratios = np.zeros(num_simulations)
    max_drawdowns = np.zeros(num_simulations)
    max_drawdown_durations = np.zeros(num_simulations)
    total_drawdown_days = np.zeros(num_simulations)

    progress_bar = st.progress(0)
    
    for i in range(num_simulations):
        if i % 1000 == 0:
            progress_bar.progress(i / num_simulations)
            
        simulated_returns = np.zeros(simulation_length)
        for j in range(simulation_length):
            if np.random.random() < prob_positive:
                simulated_returns[j] = np.random.choice(positive_returns)
            else:
                simulated_returns[j] = np.random.choice(negative_returns)

        cum_return = 0
        cum_returns = [cum_return]
        peak = 0
        max_drawdown = 0

        in_drawdown = False
        current_drawdown_duration = 0
        max_dd_duration = 0
        total_dd_days = 0

        for r in simulated_returns:
            r_decimal = r / 100.0
            cum_return = (1 + cum_return / 100) * (1 + r_decimal) * 100 - 100
            cum_returns.append(cum_return)

            if cum_return > peak:
                peak = cum_return
                if in_drawdown:
                    in_drawdown = False
                    max_dd_duration = max(max_dd_duration, current_drawdown_duration)
                    current_drawdown_duration = 0

            drawdown = ((peak - cum_return) / (1 + peak / 100)) if peak > 0 else 0

            if drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                if in_drawdown:
                    current_drawdown_duration += 1
                    total_dd_days += 1

            max_drawdown = max(max_drawdown, drawdown)

        if in_drawdown:
            max_dd_duration = max(max_dd_duration, current_drawdown_duration)

        cumulative_returns[i, :] = cum_returns

        annual_return = cum_return * (annual_periods / simulation_length)
        annual_volatility = np.std(simulated_returns) * np.sqrt(annual_periods)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        sharpe_ratios[i] = sharpe_ratio
        max_drawdowns[i] = max_drawdown
        max_drawdown_durations[i] = max_dd_duration
        total_drawdown_days[i] = total_dd_days

    progress_bar.empty()

    percentile_5 = np.percentile(cumulative_returns, 5, axis=0)
    percentile_25 = np.percentile(cumulative_returns, 25, axis=0)
    percentile_50 = np.percentile(cumulative_returns, 50, axis=0)
    percentile_75 = np.percentile(cumulative_returns, 75, axis=0)
    percentile_95 = np.percentile(cumulative_returns, 95, axis=0)

    final_returns = cumulative_returns[:, -1]

    results = {
        'final_returns': final_returns,
        'paths': cumulative_returns,
        'percentiles': {
            '5': percentile_5,
            '25': percentile_25,
            '50': percentile_50,
            '75': percentile_75,
            '95': percentile_95
        },
        'sharpe_ratios': sharpe_ratios,
        'max_drawdowns': max_drawdowns,
        'max_drawdown_durations': max_drawdown_durations,
        'total_drawdown_days': total_drawdown_days
    }

    return results

def create_simulation_plot(simulation_results, actual_path=None, title="Monte Carlo Simulation Results"):
    """Create a plot showing simulation results"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    percentiles = simulation_results['percentiles']
    x = range(len(percentiles['50']))
    
    ax.fill_between(x, percentiles['5'], percentiles['95'], 
                   color='lightblue', alpha=0.3, label='5th-95th Percentile')
    ax.fill_between(x, percentiles['25'], percentiles['75'], 
                   color='blue', alpha=0.3, label='25th-75th Percentile')
    ax.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')
    
    if actual_path is not None:
        ax.plot(x, actual_path, 'orange', linewidth=3, label='Actual Path')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    return fig

def create_distribution_plot(values, title, xlabel, actual_value=None):
    """Create a distribution plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(values, kde=True, bins=50, ax=ax)
    
    if actual_value is not None:
        ax.axvline(x=actual_value, color='r', linestyle='--', 
                  label=f'Actual: {actual_value:.2f}')
        percentile = stats.percentileofscore(values, actual_value)
        ax.text(0.05, 0.95, f'Actual Percentile: {percentile:.1f}%',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    mean_val = np.mean(values)
    median_val = np.median(values)
    
    ax.axvline(x=mean_val, color='green', linestyle='--', 
              label=f'Mean: {mean_val:.2f}')
    ax.axvline(x=median_val, color='blue', linestyle='--', 
              label=f'Median: {median_val:.2f}')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

# Main Streamlit Application
def main():
    st.title("Monte Carlo Portfolio Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Data Input",
        "Walk-Forward Analysis", 
        "Rolling Walk Tests",
        "Expanding Window Tests",
        "Forward Forecast",
        "Results Summary"
    ])
    
    # Initialize session state
    if 'returns_data' not in st.session_state:
        st.session_state.returns_data = None
    if 'portfolio_name' not in st.session_state:
        st.session_state.portfolio_name = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    if page == "Data Input":
        st.header("Portfolio Data Input")
        
        input_method = st.radio("Choose input method:", [
            "Composer Symphony URL",
            "Upload CSV File"
        ])
        
        if input_method == "Composer Symphony URL":
            default_url = 'https://app.composer.trade/symphony/BrnnCuy0Dhz3DjaAZbFt/details'
            symphony_url = st.text_input("Enter Composer Symphony URL:", value=default_url)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=date(2000, 1, 1), 
                                         help="Default: 2000-01-01 (matches original script)")
            with col2:
                end_date = st.date_input("End Date", value=date.today(),
                                       help="Default: Today's date")
            
            if st.button("Fetch Data"):
                with st.spinner("Fetching data from Composer..."):
                    allocations_df, symphony_name, tickers = fetch_backtest(
                        symphony_url, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if allocations_df is not None:
                        st.success(f"Successfully fetched data for: {symphony_name}")
                        
                        with st.spinner("Calculating portfolio returns..."):
                            daily_returns, dates = calculate_portfolio_returns(allocations_df, tickers)
                            
                            # Convert dates to strings for easier handling (matching original script)
                            date_strs = [d.strftime('%Y-%m-%d') for d in dates]
                            
                            # Check if we have enough data (matching original script validation)
                            if len(daily_returns) < 60:  # Need at least 60 days of data
                                st.error("Error: Not enough historical data for Monte Carlo analysis (minimum 60 days required).")
                                return
                            
                            # Make sure date_strs and daily_returns have the same length (matching original script)
                            if len(date_strs) != len(daily_returns):
                                st.warning(f"Warning: Length mismatch - date_strs: {len(date_strs)}, daily_returns: {len(daily_returns)}")
                                # Trim to the shorter length
                                min_length = min(len(date_strs), len(daily_returns))
                                date_strs = date_strs[:min_length]
                                # For daily_returns, we need to handle it as a pandas Series
                                if isinstance(daily_returns, pd.Series):
                                    daily_returns = daily_returns.iloc[:min_length]
                                else:
                                    daily_returns = daily_returns[:min_length]
                            
                            # Store in session state (matching original script data structure)
                            st.session_state.returns_data = {
                                'returns': daily_returns.tolist() if not isinstance(daily_returns, list) else daily_returns,
                                'dates': date_strs,
                                'name': symphony_name
                            }
                            st.session_state.portfolio_name = symphony_name
                            
                            st.success(f"Calculated returns for {len(daily_returns)} trading days")
                            
                            # Display portfolio information (matching original script output)
                            st.info(f"**Analyzing portfolio:** {symphony_name}")
                            st.info(f"**Historical data period:** {date_strs[0]} to {date_strs[-1]}")
                            st.info(f"**Total trading days:** {len(daily_returns)}")
                            
                            # Show basic statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Days", len(daily_returns))
                            with col2:
                                st.metric("Avg Daily Return", f"{np.mean(daily_returns):.4f}%")
                            with col3:
                                st.metric("Daily Volatility", f"{np.std(daily_returns):.4f}%")
                            with col4:
                                annualized_return = np.mean(daily_returns) * 252
                                st.metric("Annualized Return", f"{annualized_return:.2f}%")
        
        elif input_method == "Upload CSV File":
            st.info("Upload a CSV file with columns: 'Date' and 'Daily_Return'")
            uploaded_file = st.file_uploader("Choose CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'Date' in df.columns and 'Daily_Return' in df.columns:
                        portfolio_name = st.text_input("Portfolio Name:", 
                                                     value=uploaded_file.name.replace('.csv', ''))
                        
                        st.session_state.returns_data = {
                            'returns': df['Daily_Return'].tolist(),
                            'dates': df['Date'].tolist(),
                            'name': portfolio_name
                        }
                        st.session_state.portfolio_name = portfolio_name
                        
                        st.success(f"Successfully loaded {len(df)} trading days")
                        
                        # Show preview
                        st.subheader("Data Preview")
                        st.dataframe(df.head(10))
                        
                    else:
                        st.error("CSV must contain 'Date' and 'Daily_Return' columns")
                        
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        # Show current data status
        if st.session_state.returns_data is not None:
            st.subheader("Current Data Status")
            data = st.session_state.returns_data
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Portfolio:** {data['name']}")
            with col2:
                st.info(f"**Period:** {data['dates'][0]} to {data['dates'][-1]}")
            with col3:
                st.info(f"**Trading Days:** {len(data['returns'])}")
            
            # Plot returns
            fig, ax = plt.subplots(figsize=(12, 6))
            cumulative_returns = np.cumprod(1 + np.array(data['returns'])/100) - 1
            ax.plot(cumulative_returns * 100)
            ax.set_title(f"Cumulative Returns - {data['name']}")
            ax.set_xlabel("Trading Days")
            ax.set_ylabel("Cumulative Return (%)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Additional data visualizations
            st.subheader("Data Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily returns distribution
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                ax_hist.hist(data['returns'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax_hist.axvline(x=np.mean(data['returns']), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(data['returns']):.4f}%')
                ax_hist.set_title("Daily Returns Distribution")
                ax_hist.set_xlabel("Daily Return (%)")
                ax_hist.set_ylabel("Frequency")
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist)
            
            with col2:
                # Rolling volatility
                fig_vol, ax_vol = plt.subplots(figsize=(10, 6))
                window = min(30, len(data['returns']) // 4)
                rolling_vol = pd.Series(data['returns']).rolling(window=window).std() * np.sqrt(252)
                ax_vol.plot(rolling_vol * 100, color='orange')
                ax_vol.set_title(f"Rolling Volatility ({window}-day window)")
                ax_vol.set_xlabel("Trading Days")
                ax_vol.set_ylabel("Annualized Volatility (%)")
                ax_vol.grid(True, alpha=0.3)
                st.pyplot(fig_vol)
            
            # Key statistics
            st.subheader("Portfolio Statistics")
            returns_array = np.array(data['returns'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{cumulative_returns[-1] * 100:.2f}%")
            with col2:
                st.metric("Annualized Return", f"{np.mean(returns_array) * 252:.2f}%")
            with col3:
                st.metric("Annualized Volatility", f"{np.std(returns_array) * np.sqrt(252):.2f}%")
            with col4:
                sharpe = (np.mean(returns_array) * 252) / (np.std(returns_array) * np.sqrt(252)) if np.std(returns_array) > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            peak = 0
            drawdown = []
            for ret in cumulative_returns * 100:
                if ret > peak:
                    peak = ret
                dd = (peak - ret) / (1 + peak / 100) if peak > 0 else 0
                drawdown.append(dd)
            
            fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
            ax_dd.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
            ax_dd.plot(drawdown, color='red', linewidth=1)
            ax_dd.set_title("Portfolio Drawdown")
            ax_dd.set_xlabel("Trading Days")
            ax_dd.set_ylabel("Drawdown (%)")
            ax_dd.grid(True, alpha=0.3)
            st.pyplot(fig_dd)
            
            max_dd = max(drawdown)
            st.metric("Maximum Drawdown", f"{max_dd:.2f}%")

    elif page == "Walk-Forward Analysis":
        st.header("Walk-Forward Analysis")
        
        if st.session_state.returns_data is None:
            st.warning("Please load portfolio data first from the Data Input page.")
            return
        
        data = st.session_state.returns_data
        returns = data['returns']
        dates = data['dates']
        
        st.subheader("Test Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            # Define test periods matching original script
            available_periods = [63, 126, 252]  # 3m, 6m, 1y
            # Add 2-year test if we have enough data (matching original script logic)
            if len(returns) >= (504 + 60):  # Need at least 60 days of training data
                available_periods.append(504)
            
            test_periods = st.multiselect(
                "Select test periods (trading days):",
                available_periods,
                default=[63, 126, 252],
                help="63 days ≈ 3 months, 126 days ≈ 6 months, 252 days ≈ 1 year, 504 days ≈ 2 years"
            )
        
        with col2:
            num_simulations = st.number_input(
                "Number of simulations:",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="Default: 10,000 simulations"
            )
        
        if st.button("Run Walk-Forward Analysis"):
            results = {}
            
            for period_length in test_periods:
                if period_length >= len(returns):
                    st.warning(f"Skipping {period_length}-day test: not enough historical data")
                    continue
                
                st.subheader(f"Walk-Forward Test: {period_length} Days")
                
                # Split data
                train_returns = returns[:-period_length]
                test_returns = returns[-period_length:]
                test_dates = dates[-period_length:]
                
                with st.spinner(f"Running {num_simulations:,} simulations for {period_length} days..."):
                    simulation_results = run_monte_carlo_simulation(
                        train_returns, num_simulations, period_length
                    )
                
                # Calculate actual path
                actual_path = [0.0]
                cumulative_return = 0.0
                
                for r in test_returns:
                    r_decimal = r / 100.0
                    cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
                    actual_path.append(cumulative_return)
                
                actual_final_return = actual_path[-1]
                final_returns = simulation_results['final_returns']
                actual_percentile = stats.percentileofscore(final_returns, actual_final_return)
                
                # Store results
                results[period_length] = {
                    'actual_return': actual_final_return,
                    'forecast_return': simulation_results['percentiles']['50'][-1],
                    'percentile': actual_percentile,
                    'simulation_results': simulation_results,
                    'actual_path': actual_path,
                    'test_dates': test_dates
                }
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Actual Return", f"{actual_final_return:.2f}%")
                with col2:
                    st.metric("Forecast Return", f"{simulation_results['percentiles']['50'][-1]:.2f}%")
                with col3:
                    st.metric("Percentile Rank", f"{actual_percentile:.1f}%")
                with col4:
                    error = actual_final_return - simulation_results['percentiles']['50'][-1]
                    st.metric("Forecast Error", f"{error:.2f}%")
                
                # Create simulation plot
                fig = create_simulation_plot(
                    simulation_results, 
                    actual_path,
                    f"Walk-Forward Test: {period_length} Days ({test_dates[0]} to {test_dates[-1]})"
                )
                st.pyplot(fig)
                
                # Create distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = create_distribution_plot(
                        final_returns,
                        f"Return Distribution - {period_length} Days",
                        "Cumulative Return (%)",
                        actual_final_return
                    )
                    st.pyplot(fig_dist)
                
                with col2:
                    fig_dd = create_distribution_plot(
                        simulation_results['max_drawdowns'],
                        f"Max Drawdown Distribution - {period_length} Days",
                        "Maximum Drawdown (%)"
                    )
                    st.pyplot(fig_dd)
                
                # Generate CAGR distribution plot for longer periods
                if period_length >= 252:  # Only for 1-year or longer periods
                    years = period_length / 252
                    cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
                    actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100
                    
                    fig_cagr, ax_cagr = plt.subplots(figsize=(10, 6))
                    ax_cagr.hist(cagr_values, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                    ax_cagr.axvline(x=actual_cagr, color='r', linestyle='--', 
                                   label=f'Actual CAGR: {actual_cagr:.2f}%')
                    
                    # Add percentile information
                    cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
                    ax_cagr.text(0.05, 0.95, f'Actual CAGR Percentile: {cagr_percentile:.1f}%',
                                transform=ax_cagr.transAxes, fontsize=12,
                                verticalalignment='top', 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax_cagr.set_title(f'CAGR Distribution - {period_length} Days Forward Test', fontsize=14)
                    ax_cagr.set_xlabel('CAGR (%)', fontsize=12)
                    ax_cagr.set_ylabel('Frequency', fontsize=12)
                    ax_cagr.grid(True, alpha=0.3)
                    ax_cagr.legend()
                    st.pyplot(fig_cagr)
                    
                    # Display CAGR statistics
                    cagr_mean = np.mean(cagr_values)
                    cagr_median = np.median(cagr_values)
                    cagr_std = np.std(cagr_values)
                    cagr_5th = np.percentile(cagr_values, 5)
                    cagr_95th = np.percentile(cagr_values, 95)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Mean CAGR", f"{cagr_mean:.2f}%")
                    with col2:
                        st.metric("Median CAGR", f"{cagr_median:.2f}%")
                    with col3:
                        st.metric("CAGR Std Dev", f"{cagr_std:.2f}%")
                    with col4:
                        st.metric("5th Percentile", f"{cagr_5th:.2f}%")
                    with col5:
                        st.metric("95th Percentile", f"{cagr_95th:.2f}%")
            
            st.session_state.analysis_results['walk_forward'] = results
            
            # Summary table
            if results:
                st.subheader("Walk-Forward Test Summary")
                
                summary_data = []
                for period, result in results.items():
                    summary_data.append({
                        'Period (Days)': period,
                        'Actual Return (%)': f"{result['actual_return']:.2f}",
                        'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                        'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                        'Percentile Rank': f"{result['percentile']:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    elif page == "Rolling Walk Tests":
        st.header("Rolling Walk-Forward Tests")
        
        if st.session_state.returns_data is None:
            st.warning("Please load portfolio data first from the Data Input page.")
            return
        
        data = st.session_state.returns_data
        returns = data['returns']
        dates = data['dates']
        
        st.subheader("Test Configuration")
        
        test_type = st.radio("Test Type:", [
            "Sliding Window", 
            "Fixed Training Window"
        ])
        
        col1, col2, col3 = st.columns(3)
        
        if test_type == "Sliding Window":
            with col1:
                train_length = st.number_input("Training Period (days):", 
                                             min_value=60, max_value=len(returns)//2, 
                                             value=252,
                                             help="Default: 252 days (1 year)")
            with col2:
                test_length = st.number_input("Test Period (days):", 
                                            min_value=20, max_value=504, 
                                            value=252,
                                            help="Default: 252 days (1 year)")
            with col3:
                step_size = st.number_input("Step Size (days):", 
                                          min_value=1, max_value=test_length, 
                                          value=63,
                                          help="Default: 63 days (quarterly)")
            
            allow_overlap = st.checkbox("Allow test windows to overlap with training data")
            
        else:  # Fixed Training Window
            with col1:
                train_start = st.date_input("Training Start:", 
                                          value=pd.to_datetime(dates[0]))
            with col2:
                train_end = st.date_input("Training End:", 
                                        value=pd.to_datetime(dates[len(dates)//2]))
            with col3:
                test_length = st.number_input("Test Period (days):", 
                                            min_value=20, max_value=504, 
                                            value=252,
                                            help="Default: 252 days (1 year)")
            
            step_size = st.number_input("Step Size (days):", 
                                      min_value=1, max_value=test_length, 
                                      value=63,
                                      help="Default: 63 days (quarterly)")
            allow_overlap = st.checkbox("Allow test windows to overlap with training data")
        
        num_simulations = st.number_input("Number of simulations:", 
                                        min_value=1000, max_value=20000, 
                                        value=10000,
                                        help="Default: 10,000 simulations")
        
        if st.button("Run Rolling Walk Tests"):
            if test_type == "Sliding Window":
                if len(returns) < (train_length + test_length) and not allow_overlap:
                    st.error(f"Not enough data for non-overlapping test. Need at least {train_length + test_length} days.")
                    return
                
                available_test_days = len(returns) - (0 if allow_overlap else train_length)
                num_iterations = max(1, (available_test_days - test_length) // step_size + 1)
                
                st.info(f"Running {num_iterations} iterations with sliding window approach")
                
                results = []
                progress_bar = st.progress(0)
                
                for i in range(num_iterations):
                    progress_bar.progress((i + 1) / num_iterations)
                    
                    if allow_overlap:
                        test_start_idx = i * step_size
                        train_start_idx = max(0, test_start_idx - train_length)
                        train_end_idx = test_start_idx
                    else:
                        train_start_idx = i * step_size
                        train_end_idx = train_start_idx + train_length
                        test_start_idx = train_end_idx
                    
                    test_end_idx = min(test_start_idx + test_length, len(returns))
                    
                    if test_end_idx - test_start_idx < 20:
                        continue
                    
                    train_data = returns[train_start_idx:train_end_idx]
                    test_data = returns[test_start_idx:test_end_idx]
                    
                    simulation_results = run_monte_carlo_simulation(
                        train_data, num_simulations, len(test_data)
                    )
                    
                    # Calculate actual path
                    actual_path = [0.0]
                    cumulative_return = 0.0
                    
                    for r in test_data:
                        r_decimal = r / 100.0
                        cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
                        actual_path.append(cumulative_return)
                    
                    actual_final_return = actual_path[-1]
                    final_returns = simulation_results['final_returns']
                    percentile = stats.percentileofscore(final_returns, actual_final_return)
                    
                    results.append({
                        'iteration': i + 1,
                        'test_start_date': dates[test_start_idx],
                        'test_end_date': dates[test_end_idx - 1],
                        'actual_return': actual_final_return,
                        'forecast_return': simulation_results['percentiles']['50'][-1],
                        'percentile': percentile,
                        'max_drawdown': np.mean(simulation_results['max_drawdowns']) # Use mean of max_drawdowns for rolling
                    })
                
                st.session_state.analysis_results['rolling_walk'] = results
                
                # Summary table
                if results:
                    st.subheader("Rolling Walk-Forward Test Summary")
                    
                    summary_data = []
                    for result in results:
                        summary_data.append({
                            'Iteration': result['iteration'],
                            'Test Period': f"{result['test_start_date']} to {result['test_end_date']}",
                            'Actual Return (%)': f"{result['actual_return']:.2f}",
                            'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                            'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                            'Percentile Rank': f"{result['percentile']:.1f}%",
                            'Max Drawdown (%)': f"{result['max_drawdown']:.2f}"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                    
                    # Create aggregate visualizations
                    st.subheader("Rolling Test Analysis")
                    
                    # Plot forecast accuracy over time
                    fig, ax = plt.subplots(figsize=(12, 6))
                    iterations = [r['iteration'] for r in results]
                    actual_returns = [r['actual_return'] for r in results]
                    forecast_returns = [r['forecast_return'] for r in results]
                    errors = [r['actual_return'] - r['forecast_return'] for r in results]
                    
                    ax.plot(iterations, actual_returns, 'o-', label='Actual Returns', color='blue')
                    ax.plot(iterations, forecast_returns, 's-', label='Forecast Returns', color='red')
                    ax.fill_between(iterations, [a-e for a, e in zip(actual_returns, errors)], 
                                  [a+e for a, e in zip(actual_returns, errors)], 
                                  alpha=0.2, color='gray', label='Error Range')
                    
                    ax.set_title("Rolling Walk-Forward Test Results")
                    ax.set_xlabel("Test Iteration")
                    ax.set_ylabel("Return (%)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Plot percentile ranks over time
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    percentiles = [r['percentile'] for r in results]
                    ax2.plot(iterations, percentiles, 'o-', color='green')
                    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50th Percentile')
                    ax2.set_title("Percentile Ranks Over Time")
                    ax2.set_xlabel("Test Iteration")
                    ax2.set_ylabel("Percentile Rank")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    # Error distribution
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    ax3.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax3.axvline(x=np.mean(errors), color='red', linestyle='--', 
                               label=f'Mean Error: {np.mean(errors):.2f}%')
                    ax3.set_title("Forecast Error Distribution")
                    ax3.set_xlabel("Forecast Error (%)")
                    ax3.set_ylabel("Frequency")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                    st.pyplot(fig3)

    elif page == "Expanding Window Tests":
        st.header("Expanding Window Tests")
        
        if st.session_state.returns_data is None:
            st.warning("Please load portfolio data first from the Data Input page.")
            return
        
        data = st.session_state.returns_data
        returns = data['returns']
        dates = data['dates']
        
        st.subheader("Test Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            initial_train_period = st.number_input(
                "Initial Training Period (days):",
                min_value=60,
                max_value=len(returns)//2,
                value=252,
                help="Default: 252 days (1 year) - Length of initial training window"
            )
        
        with col2:
            test_period_length = st.number_input(
                "Test Period Length (days):",
                min_value=20,
                max_value=504,
                value=252,
                help="Default: 252 days (1 year) - Length of each test period"
            )
        
        with col3:
            expansion_size = st.number_input(
                "Expansion Size (days):",
                min_value=20,
                max_value=252,
                value=252,
                help="Default: 252 days (1 year) - Days to expand training window by each iteration"
            )
        
        num_simulations = st.number_input(
            "Number of simulations:",
            min_value=1000,
            max_value=20000,
            value=10000,
            help="Default: 10,000 simulations"
        )
        
        if st.button("Run Expanding Window Tests"):
            if len(returns) < (initial_train_period + test_period_length):
                st.error(f"Not enough data for expanding window test. Need at least {initial_train_period + test_period_length} days.")
                return
            
            # Determine number of iterations
            available_days = len(returns) - initial_train_period
            num_iterations = available_days // test_period_length
            
            if num_iterations == 0:
                st.error("Not enough data for any complete test periods after initial training window.")
                return
            
            st.info(f"Running {num_iterations} iterations with expanding window approach")
            
            results = []
            progress_bar = st.progress(0)
            
            for i in range(num_iterations):
                progress_bar.progress((i + 1) / num_iterations)
                
                # Calculate indices for this iteration
                train_start_idx = 0  # Always start from beginning
                train_size = initial_train_period + (i * expansion_size)
                train_end_idx = train_size
                test_start_idx = train_end_idx
                test_end_idx = min(test_start_idx + test_period_length, len(returns))
                
                if test_end_idx - test_start_idx < 5:
                    continue
                
                # Extract data
                train_data = returns[train_start_idx:train_end_idx]
                test_data = returns[test_start_idx:test_end_idx]
                test_dates = dates[test_start_idx:test_end_idx]
                
                train_start_date = dates[train_start_idx]
                train_end_date = dates[train_end_idx - 1]
                test_start_date = dates[test_start_idx]
                test_end_date = dates[test_end_idx - 1]
                
                # Run simulation
                simulation_results = run_monte_carlo_simulation(
                    train_data, num_simulations, len(test_data)
                )
                
                # Calculate actual path
                actual_path = [0.0]
                cumulative_return = 0.0
                
                for r in test_data:
                    r_decimal = r / 100.0
                    cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
                    actual_path.append(cumulative_return)
                
                actual_final_return = actual_path[-1]
                final_returns = simulation_results['final_returns']
                percentile = stats.percentileofscore(final_returns, actual_final_return)
                
                # Calculate CAGR for longer periods
                if len(test_data) >= 252:
                    years = len(test_data) / 252
                    actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100
                    cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
                    cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
                else:
                    actual_cagr = actual_final_return
                    cagr_percentile = percentile
                
                results.append({
                    'iteration': i + 1,
                    'train_size': train_size,
                    'train_period': f"{train_start_date} to {train_end_date}",
                    'test_period': f"{test_start_date} to {test_end_date}",
                    'actual_return': actual_final_return,
                    'forecast_return': simulation_results['percentiles']['50'][-1],
                    'percentile': percentile,
                    'actual_cagr': actual_cagr,
                    'cagr_percentile': cagr_percentile,
                    'max_drawdown': np.mean(simulation_results['max_drawdowns'])
                })
            
            st.session_state.analysis_results['expanding_window'] = results
            
            # Summary table
            if results:
                st.subheader("Expanding Window Test Summary")
                
                summary_data = []
                for result in results:
                    summary_data.append({
                        'Iteration': result['iteration'],
                        'Train Size': result['train_size'],
                        'Test Period': result['test_period'],
                        'Actual Return (%)': f"{result['actual_return']:.2f}",
                        'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                        'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                        'Percentile Rank': f"{result['percentile']:.1f}%",
                        'CAGR (%)': f"{result['actual_cagr']:.2f}",
                        'CAGR Percentile': f"{result['cagr_percentile']:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                # Visualizations
                st.subheader("Expanding Window Analysis")
                
                # Plot training size vs performance
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                train_sizes = [r['train_size'] for r in results]
                actual_returns = [r['actual_return'] for r in results]
                forecast_returns = [r['forecast_return'] for r in results]
                
                ax1.plot(train_sizes, actual_returns, 'o-', label='Actual Returns', color='blue')
                ax1.plot(train_sizes, forecast_returns, 's-', label='Forecast Returns', color='red')
                ax1.set_title("Performance vs Training Window Size")
                ax1.set_xlabel("Training Window Size (days)")
                ax1.set_ylabel("Return (%)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # Plot percentile ranks over time
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                iterations = [r['iteration'] for r in results]
                percentiles = [r['percentile'] for r in results]
                cagr_percentiles = [r['cagr_percentile'] for r in results]
                
                ax2.plot(iterations, percentiles, 'o-', label='Return Percentile', color='green')
                ax2.plot(iterations, cagr_percentiles, 's-', label='CAGR Percentile', color='orange')
                ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50th Percentile')
                ax2.set_title("Percentile Ranks Over Time")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Percentile Rank")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
                
                # Error analysis
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                errors = [r['actual_return'] - r['forecast_return'] for r in results]
                ax3.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=np.mean(errors), color='red', linestyle='--', 
                           label=f'Mean Error: {np.mean(errors):.2f}%')
                ax3.set_title("Forecast Error Distribution")
                ax3.set_xlabel("Forecast Error (%)")
                ax3.set_ylabel("Frequency")
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)

    elif page == "Forward Forecast":
        st.header("Forward Forecast")
        
        if st.session_state.returns_data is None:
            st.warning("Please load portfolio data first from the Data Input page.")
            return
        
        data = st.session_state.returns_data
        returns = data['returns']
        
        st.subheader("Forecast Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_days = st.number_input(
                "Forecast period (trading days):",
                min_value=30,
                max_value=1000,
                value=126,
                help="Default: 126 days (~6 months) - Number of trading days to forecast"
            )
        
        with col2:
            num_simulations = st.number_input(
                "Number of simulations:",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="Default: 10,000 simulations"
            )
        
        if st.button("Run Forward Forecast"):
            with st.spinner(f"Running {num_simulations:,} simulations for {forecast_days} days..."):
                simulation_results = run_monte_carlo_simulation(
                    returns, num_simulations, forecast_days
                )
            
            st.subheader("Forecast Results")
            
            # Key metrics
            final_returns = simulation_results['final_returns']
            percentiles = simulation_results['percentiles']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Median Forecast", f"{np.median(final_returns):.2f}%")
            with col2:
                st.metric("5th Percentile", f"{np.percentile(final_returns, 5):.2f}%")
            with col3:
                st.metric("95th Percentile", f"{np.percentile(final_returns, 95):.2f}%")
            with col4:
                st.metric("Probability of Loss", f"{np.mean(final_returns < 0) * 100:.1f}%")
            
            # Create enhanced simulation plot with sample paths
            fig, ax = plt.subplots(figsize=(12, 8))
            x = range(len(percentiles['50']))
            
            # Select random sample paths (matching original script default of 200)
            all_paths = simulation_results['paths']
            num_paths = all_paths.shape[0]
            num_samples = min(200, num_paths)  # Default from original script
            sample_indices = np.random.choice(num_paths, num_samples, replace=False)
            
            # Plot random sample paths with very light opacity
            for idx in sample_indices:
                path = all_paths[idx, :]
                ax.plot(x, path, color='gray', alpha=0.3, linewidth=0.5)
            
            # Plot percentile bands
            ax.fill_between(x, percentiles['5'], percentiles['95'], 
                           color='lightblue', alpha=0.3, label='5th-95th Percentile')
            ax.fill_between(x, percentiles['25'], percentiles['75'], 
                           color='blue', alpha=0.3, label='25th-75th Percentile')
            ax.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Forecast')
            
            # Add sample paths label
            ax.plot([], [], color='gray', alpha=0.5, linewidth=1, label=f'{num_samples} Sample Paths')
            
            # Calculate approximate end date
            import datetime as dt
            last_date = dt.datetime.strptime(dates[-1], '%Y-%m-%d')
            forecast_calendar_days = int(forecast_days * 1.4)
            forecast_end_date = (last_date + dt.timedelta(days=forecast_calendar_days)).strftime('%Y-%m-%d')
            
            ax.set_title(f'Forward Return Forecast: {forecast_days} trading days\n({dates[-1]} to ~{forecast_end_date})', 
                        fontsize=14)
            ax.set_xlabel('Trading Days', fontsize=12)
            ax.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Add key metrics as text
            final_50th = percentiles['50'][-1]
            final_5th = percentiles['5'][-1]
            final_25th = percentiles['25'][-1]
            final_75th = percentiles['75'][-1]
            final_95th = percentiles['95'][-1]
            
            info_text = (f"Median Forecast: {final_50th:.2f}%\n"
                        f"5th-95th Range: {final_5th:.2f}% to {final_95th:.2f}%\n"
                        f"25th-75th Range: {final_25th:.2f}% to {final_75th:.2f}%\n"
                        f"Expected Max Drawdown: {np.mean(simulation_results['max_drawdowns']):.2f}%")
            
            # Add CAGR if forecast period is meaningful
            if forecast_days >= 60:
                years = forecast_days / 252
                cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
                cagr_median = np.median(cagr_values)
                cagr_5th = np.percentile(cagr_values, 5)
                cagr_95th = np.percentile(cagr_values, 95)
                info_text += f"\n\nMedian CAGR: {cagr_median:.2f}%\n5th-95th CAGR: {cagr_5th:.2f}% to {cagr_95th:.2f}%"
            
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', horizontalalignment='right')
            
            st.pyplot(fig)
            
            # Create distribution plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced return distribution plot
                fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
                sns.histplot(final_returns, kde=True, bins=50, color='blue', ax=ax_dist)
                
                # Add vertical lines for key percentiles
                final_5th = np.percentile(final_returns, 5)
                final_25th = np.percentile(final_returns, 25)
                final_50th = np.percentile(final_returns, 50)
                final_75th = np.percentile(final_returns, 75)
                final_95th = np.percentile(final_returns, 95)
                
                ax_dist.axvline(x=final_5th, color='red', linestyle='--', alpha=0.7, label=f'5th: {final_5th:.2f}%')
                ax_dist.axvline(x=final_50th, color='green', linestyle='--', linewidth=2, label=f'Median: {final_50th:.2f}%')
                ax_dist.axvline(x=final_95th, color='purple', linestyle='--', alpha=0.7, label=f'95th: {final_95th:.2f}%')
                
                ax_dist.set_title(f'Return Distribution - {forecast_days} Day Forward Forecast', fontsize=14)
                ax_dist.set_xlabel('Cumulative Return (%)', fontsize=12)
                ax_dist.set_ylabel('Frequency', fontsize=12)
                ax_dist.grid(True, alpha=0.3)
                ax_dist.legend()
                st.pyplot(fig_dist)
            
            with col2:
                fig_dd = create_distribution_plot(
                    simulation_results['max_drawdowns'],
                    f"Max Drawdown Distribution - {forecast_days} Days",
                    "Maximum Drawdown (%)"
                )
                st.pyplot(fig_dd)
            
            st.session_state.analysis_results['forward_forecast'] = simulation_results

    elif page == "Results Summary":
        st.header("Analysis Results Summary")
        
        if not st.session_state.analysis_results:
            st.info("No analysis results available. Run analyses from other pages first.")
            return
        
        # Summary of all available results
        for analysis_type, results in st.session_state.analysis_results.items():
            st.subheader(f"{analysis_type.replace('_', ' ').title()}")
            
            if analysis_type == 'walk_forward':
                if results:
                    summary_data = []
                    for period, result in results.items():
                        summary_data.append({
                            'Period (Days)': period,
                            'Actual Return (%)': f"{result['actual_return']:.2f}",
                            'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                            'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                            'Percentile Rank': f"{result['percentile']:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            elif analysis_type == 'rolling_walk':
                if results:
                    summary_data = []
                    for result in results:
                        summary_data.append({
                            'Iteration': result['iteration'],
                            'Test Period': f"{result['test_start_date']} to {result['test_end_date']}",
                            'Actual Return (%)': f"{result['actual_return']:.2f}",
                            'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                            'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                            'Percentile Rank': f"{result['percentile']:.1f}%",
                            'Max Drawdown (%)': f"{result['max_drawdown']:.2f}"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            elif analysis_type == 'expanding_window':
                if results:
                    summary_data = []
                    for result in results:
                        summary_data.append({
                            'Iteration': result['iteration'],
                            'Train Size': result['train_size'],
                            'Test Period': result['test_period'],
                            'Actual Return (%)': f"{result['actual_return']:.2f}",
                            'Forecast Return (%)': f"{result['forecast_return']:.2f}",
                            'Error (%)': f"{result['actual_return'] - result['forecast_return']:.2f}",
                            'Percentile Rank': f"{result['percentile']:.1f}%",
                            'CAGR (%)': f"{result['actual_cagr']:.2f}",
                            'CAGR Percentile': f"{result['cagr_percentile']:.1f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            elif analysis_type == 'forward_forecast':
                final_returns = results['final_returns']
                st.write(f"**Forecast Statistics:**")
                st.write(f"- Median: {np.median(final_returns):.2f}%")
                st.write(f"- 5th Percentile: {np.percentile(final_returns, 5):.2f}%")
                st.write(f"- 95th Percentile: {np.percentile(final_returns, 95):.2f}%")
                st.write(f"- Probability of Loss: {np.mean(final_returns < 0) * 100:.1f}%")
                
                # Create summary visualization
                fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
                ax_summary.hist(final_returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
                ax_summary.axvline(x=np.median(final_returns), color='red', linestyle='--', 
                                  label=f'Median: {np.median(final_returns):.2f}%')
                ax_summary.axvline(x=np.percentile(final_returns, 5), color='orange', linestyle='--', 
                                  label=f'5th Percentile: {np.percentile(final_returns, 5):.2f}%')
                ax_summary.axvline(x=np.percentile(final_returns, 95), color='green', linestyle='--', 
                                  label=f'95th Percentile: {np.percentile(final_returns, 95):.2f}%')
                ax_summary.set_title("Forward Forecast Distribution Summary")
                ax_summary.set_xlabel("Cumulative Return (%)")
                ax_summary.set_ylabel("Frequency")
                ax_summary.legend()
                ax_summary.grid(True, alpha=0.3)
                st.pyplot(fig_summary)

if __name__ == "__main__":
    main()
