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
import random

# Set page config
st.set_page_config(
    page_title="Monte Carlo Symphony Analysis",
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

def analyze_drawdowns_comprehensive(returns, dates, period_length, test_start_date, test_end_date, portfolio_name):
    """
    Comprehensive drawdown analysis matching the original script functionality
    
    Parameters:
    -----------
    returns : list
        List of cumulative returns
    dates : list
        List of date strings corresponding to returns data
    period_length : int
        Length of the test period in days
    test_start_date : str
        Start date of the test period
    test_end_date : str
        End date of the test period
    portfolio_name : str
        Name of the portfolio for file naming
    """
    # Ensure dates is the same length as returns
    if len(dates) != len(returns):
        st.warning(f"Length mismatch between dates ({len(dates)}) and returns ({len(returns)})")
        min_length = min(len(dates), len(returns))
        dates = dates[:min_length]
        returns = returns[:min_length]

    # Convert dates to datetime objects for proper comparison
    date_objects = [pd.to_datetime(d).date() if isinstance(d, str) else d for d in dates]

    # Calculate drawdown periods and durations
    drawdown_periods = []
    drawdowns = []

    # Track the running peak
    running_peak = returns[0]
    current_drawdown_start_idx = None
    current_drawdown_start_value = None
    in_drawdown = False
    max_drawdown = 0
    max_drawdown_idx = 0

    # Loop through returns to calculate drawdowns and identify periods
    for i, value in enumerate(returns):
        # Update the running peak if we have a new high
        if value > running_peak:
            running_peak = value

            # If we were in a drawdown and now we've reached a new peak, the drawdown is over
            if in_drawdown:
                # Calculate the depth (percentage) of this drawdown period
                min_value = min(returns[current_drawdown_start_idx:i])
                drawdown_depth = ((running_peak - min_value) / (1 + running_peak / 100))

                # Add this drawdown period to our list
                drawdown_periods.append({
                    'start_idx': current_drawdown_start_idx,
                    'end_idx': i,
                    'start_date': dates[current_drawdown_start_idx],
                    'end_date': dates[i],
                    'duration': i - current_drawdown_start_idx,
                    'max_drawdown': drawdown_depth,
                    'calendar_days': (date_objects[i] - date_objects[current_drawdown_start_idx]).days
                })

                # Reset drawdown tracking
                in_drawdown = False
                current_drawdown_start_idx = None
                current_drawdown_start_value = None

        # Calculate current drawdown from peak
        current_drawdown = ((running_peak - value) / (1 + running_peak / 100))
        drawdowns.append(current_drawdown)

        # Update maximum drawdown if this is a new max
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_idx = i

        # Detect the start of a new drawdown
        if current_drawdown > 0 and not in_drawdown:
            in_drawdown = True
            current_drawdown_start_idx = i
            current_drawdown_start_value = running_peak

    # If we're still in a drawdown at the end, add that period
    if in_drawdown:
        min_value = min(returns[current_drawdown_start_idx:])
        drawdown_depth = ((running_peak - min_value) / (1 + running_peak / 100))

        drawdown_periods.append({
            'start_idx': current_drawdown_start_idx,
            'end_idx': len(returns) - 1,
            'start_date': dates[current_drawdown_start_idx],
            'end_date': dates[-1],
            'duration': len(returns) - current_drawdown_start_idx,
            'max_drawdown': drawdown_depth,
            'calendar_days': (date_objects[-1] - date_objects[current_drawdown_start_idx]).days
        })

    # Recalculate max_drawdown from the drawdowns list to ensure consistency
    if drawdowns:
        max_drawdown = max(drawdowns)
        max_drawdown_idx = drawdowns.index(max_drawdown)

    # Recalculate drawdown periods to ensure they have the correct max_drawdown value
    for i, period in enumerate(drawdown_periods):
        start_idx = period['start_idx']
        end_idx = period['end_idx']
        period_drawdowns = drawdowns[start_idx:end_idx + 1]
        if period_drawdowns:
            period_max_dd = max(period_drawdowns)
            drawdown_periods[i]['max_drawdown'] = period_max_dd

    # Sort drawdown periods by max_drawdown (descending) for proper ranking
    drawdown_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)

    total_days_in_drawdown = sum(period['duration'] for period in drawdown_periods)
    significant_drawdown_days = sum(period['calendar_days'] for period in drawdown_periods
                                    if period['calendar_days'] > 20)

    # Find significant drawdown periods (duration > 20 calendar days)
    significant_periods = [p for p in drawdown_periods if p['calendar_days'] > 20]
    significant_periods.sort(key=lambda x: x['max_drawdown'], reverse=True)

    # Display top 5 significant drawdown periods
    st.subheader("Top 5 Significant Drawdown Periods (>20 calendar days)")
    
    if significant_periods:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write("**Rank**")
        with col2:
            st.write("**Trading Days**")
        with col3:
            st.write("**Calendar Days**")
        with col4:
            st.write("**Max Drawdown**")
        with col5:
            st.write("**Period**")
        
        for i, period in enumerate(significant_periods[:5], 1):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write(f"{i}")
            with col2:
                st.write(f"{period['duration']}")
            with col3:
                st.write(f"{period['calendar_days']}")
            with col4:
                st.write(f"{period['max_drawdown']:.2f}%")
            with col5:
                st.write(f"{period['start_date']} to {period['end_date']}")
    else:
        st.info("No significant drawdown periods (>20 calendar days) found.")

    # Calculate average statistics
    avg_drawdown_length = total_days_in_drawdown / len(drawdown_periods) if drawdown_periods else 0
    avg_calendar_days = sum(period['calendar_days'] for period in drawdown_periods) / len(
        drawdown_periods) if drawdown_periods else 0
    non_zero_drawdowns = [d for d in drawdowns if d > 0]
    avg_drawdown_depth = sum(non_zero_drawdowns) / len(non_zero_drawdowns) if non_zero_drawdowns else 0

    # Display summary statistics
    st.subheader("Drawdown Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Max Drawdown", f"{max_drawdown:.2f}%")
    with col2:
        st.metric("Average Drawdown", f"{avg_drawdown_depth:.2f}%")
    with col3:
        st.metric("Total Drawdown Days", f"{total_days_in_drawdown}")
    with col4:
        st.metric("Significant Periods", f"{len(significant_periods)}")

    # Create comprehensive drawdown visualizations
    st.subheader("Drawdown Analysis Visualizations")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)

    # Create underwater plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(range(len(drawdowns)), drawdowns, 0, color='red', alpha=0.3)
    ax1.plot(range(len(drawdowns)), drawdowns, color='red', linewidth=1)

    # Customize underwater plot
    ax1.set_title(f'Drawdown Over Time - {period_length} days')
    ax1.set_ylabel('Drawdown (%)')
    ax1.set_xlabel('Trading Days')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))

    # Add maximum drawdown line and annotation
    ax1.axhline(y=max_drawdown, color='darkred', linestyle='--', alpha=0.5)
    ax1.annotate(f'Maximum Drawdown: {max_drawdown:.2f}%\n'
                 f'Average Drawdown: {avg_drawdown_depth:.2f}%\n'
                 f'Avg Trading Days: {avg_drawdown_length:.1f}\n'
                 f'Avg Calendar Days: {avg_calendar_days:.1f}',
                 xy=(max_drawdown_idx, max_drawdown),
                 xytext=(10, 10),
                 textcoords='offset points',
                 ha='left',
                 va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='red', alpha=0.8))

    # Create cumulative returns plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(returns)), returns, color='blue', linewidth=1.5)

    # Mark drawdown periods
    for period in drawdown_periods:
        ax2.axvspan(period['start_idx'], period['end_idx'],
                    alpha=0.2, color='red')

    # Customize cumulative returns plot
    ax2.set_title(f'Cumulative Return with Drawdown Periods - {period_length} days')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.set_xlabel('Trading Days')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdown durations (middle row)
    ax3 = fig.add_subplot(gs[1, 0])

    durations = [period['duration'] for period in drawdown_periods]
    calendar_days = [period['calendar_days'] for period in drawdown_periods]
    max_drawdowns = [period['max_drawdown'] for period in drawdown_periods]

    if durations:
        # Create a grouped bar chart with both trading days and calendar days
        x = np.arange(len(durations))
        width = 0.35

        ax3.bar(x - width / 2, durations, width, color='blue', alpha=0.7, label='Trading Days')
        ax3.bar(x + width / 2, calendar_days, width, color='green', alpha=0.7, label='Calendar Days')

        # Add drawdown depth as text
        for i, (_, cal_days, dd) in enumerate(zip(durations, calendar_days, max_drawdowns)):
            ax3.text(i, cal_days + 1, f"{dd:.1f}%",
                     ha='center', va='bottom',
                     color='black', fontsize=8)

        # Customize drawdown durations plot
        ax3.set_title('Drawdown Duration by Episode')
        ax3.set_ylabel('Duration (Days)')
        ax3.set_xlabel('Drawdown Episode')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 1])

    if drawdowns:
        non_zero_drawdowns = [d for d in drawdowns if d > 0]
        if non_zero_drawdowns:
            sns.histplot(non_zero_drawdowns, bins=20, kde=True, ax=ax4, color='green')

            # Add vertical line for mean and maximum
            mean_dd = np.mean(non_zero_drawdowns)
            median_dd = np.median(non_zero_drawdowns)

            ax4.axvline(mean_dd, color='red', linestyle='--',
                        label=f'Mean: {mean_dd:.2f}%')
            ax4.axvline(median_dd, color='blue', linestyle='--',
                        label=f'Median: {median_dd:.2f}%')
            ax4.axvline(max_drawdown, color='black', linestyle='-',
                        label=f'Max: {max_drawdown:.2f}%')

            # Customize drawdown distribution plot
            ax4.set_title('Drawdown Magnitude Distribution')
            ax4.set_xlabel('Drawdown (%)')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, linestyle='--', alpha=0.7)
        else:
            ax4.text(0.5, 0.5, 'No non-zero drawdowns found',
                     ha='center', va='center', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No drawdowns found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown duration distribution (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])

    if durations:
        # Plot both trading days and calendar days as separate histograms
        sns.histplot(durations, bins=min(20, len(durations)), kde=True, ax=ax5, color='blue',
                     alpha=0.4, label='Trading Days')
        sns.histplot(calendar_days, bins=min(20, len(calendar_days)), kde=True, ax=ax5, color='green',
                     alpha=0.4, label='Calendar Days')

        # Add vertical line for mean and maximum of calendar days
        mean_duration = np.mean(durations)
        mean_calendar = np.mean(calendar_days)
        median_duration = np.median(durations)
        median_calendar = np.median(calendar_days)
        max_calendar = max(calendar_days)

        ax5.axvline(mean_calendar, color='darkgreen', linestyle='--',
                    label=f'Mean Calendar: {mean_calendar:.1f} days')
        ax5.axvline(median_calendar, color='green', linestyle=':',
                    label=f'Median Calendar: {median_calendar:.1f} days')
        ax5.axvline(max_calendar, color='green', linestyle='-',
                    label=f'Max Calendar: {max_calendar:.0f} days')

        # Customize drawdown duration distribution plot
        ax5.set_title('Drawdown Duration Distribution')
        ax5.set_xlabel('Duration (Days)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, linestyle='--', alpha=0.7)
    else:
        ax5.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Plot drawdown scatter (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])

    if durations and max_drawdowns:
        # Scatter plot with calendar days instead of trading days
        ax6.scatter(max_drawdowns, calendar_days, alpha=0.7, c='green', s=50, label='Calendar Days')
        ax6.scatter(max_drawdowns, durations, alpha=0.7, c='blue', s=30, label='Trading Days')

        # Add regression line for calendar days
        if len(calendar_days) > 1:
            z = np.polyfit(max_drawdowns, calendar_days, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(max_drawdowns), max(max_drawdowns), 100)
            ax6.plot(x_range, p(x_range), 'g--', alpha=0.7)

            # Calculate correlation
            corr = np.corrcoef(max_drawdowns, calendar_days)[0, 1]
            ax6.text(0.05, 0.95, f'Calendar Day Correlation: {corr:.2f}',
                     transform=ax6.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Customize scatter plot
        ax6.set_title('Drawdown Magnitude vs Duration')
        ax6.set_xlabel('Maximum Drawdown (%)')
        ax6.set_ylabel('Duration (Days)')
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'No drawdown periods found',
                 ha='center', va='center', fontsize=12)

    # Add overall title
    fig.suptitle(f'Drawdown Analysis ({test_start_date} to {test_end_date})', fontsize=16, y=0.99)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)
    fig.subplots_adjust(top=0.92)

    st.pyplot(fig)

    # Return drawdown statistics
    return {
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown_depth,
        'total_drawdown_days': total_days_in_drawdown,
        'significant_drawdown_days': significant_drawdown_days,
        'avg_drawdown_length': avg_drawdown_length,
        'avg_calendar_days': avg_calendar_days,
        'drawdown_periods': len(drawdown_periods),
        'significant_periods': len(significant_periods),
        'max_drawdown_duration': max(period['duration'] for period in drawdown_periods) if drawdown_periods else 0,
        'max_calendar_duration': max(period['calendar_days'] for period in drawdown_periods) if drawdown_periods else 0,
        'drawdown_durations': [period['duration'] for period in drawdown_periods],
        'calendar_durations': [period['calendar_days'] for period in drawdown_periods],
        'drawdown_magnitudes': [period['max_drawdown'] for period in drawdown_periods],
        'top_significant_periods': significant_periods[:5]
    }

# Main Streamlit Application
def main():
    st.title("Monte Carlo Symphony Analysis")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Configuration",
        "Walk-Forward Analysis", 
        "Rolling Walk Tests",
        "Expanding Window Tests",
        "Forward Forecast",
        "Credits"
    ])
    
    # Initialize session state
    if 'returns_data' not in st.session_state:
        st.session_state.returns_data = None
    if 'portfolio_name' not in st.session_state:
        st.session_state.portfolio_name = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    if page == "Configuration":
        st.header("Configuration")
        
        # Handle URL parameters for shareable links
        symphony_url_param = st.query_params.get("symphony", "")
        start_date_param = st.query_params.get("start_date", None)
        end_date_param = st.query_params.get("end_date", None)
        
        # Set default values from URL parameters if available
        default_start_date = date.fromisoformat(start_date_param) if start_date_param else date(2000, 1, 1)
        default_end_date = date.fromisoformat(end_date_param) if end_date_param else date.today()
        
        input_method = "Composer Symphony URL"
        
        # Use a form like iota calculator for consistent button styling
        with st.form("fetch_data_form"):
            symphony_url = st.text_input(
                "Enter Composer Symphony URL:", 
                value=symphony_url_param,
                help="Enter the full URL of your Composer symphony",
                placeholder="https://app.composer.trade/symphony/..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=default_start_date, 
                                         help="Default: 2000-01-01 (defaults to oldest possible date)")
            with col2:
                end_date = st.date_input("End Date", value=default_end_date,
                                       help="Default: Today's date")
            
            # Submit button with same styling as iota calculator
            submitted = st.form_submit_button("Fetch Data", type="primary")
            
            if submitted:
                # Generate shareable link
                if symphony_url and symphony_url.strip():
                    # Validate that it's a Composer symphony URL
                    import re
                    symphony_id_match = re.search(r'/symphony/([^/]+)/', symphony_url)
                    if symphony_id_match:
                        symphony_id = symphony_id_match.group(1)
                        
                        # Create shareable URL with full base URL
                        base_url = "https://stochastic.streamlit.app/"
                        shareable_url = f"{base_url}?symphony={symphony_url}&start_date={start_date.strftime('%Y-%m-%d')}&end_date={end_date.strftime('%Y-%m-%d')}"
                        
                        # Store the shareable URL in session state for display outside the form
                        st.session_state.shareable_url = shareable_url
                        
                        # Success message
                        st.success("Configuration saved! Fetching data...")
                
                # Fetch data
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
        
        # Display shareable URL outside the form (if available)
        if hasattr(st.session_state, 'shareable_url') and st.session_state.shareable_url:
            st.markdown("---")
            st.markdown("### 🔗 Share Your Analysis")
            st.info("**Shareable URL**: Copy this link to share your analysis configuration with others:")
            
            # Create a text area for easy copying
            st.text_area(
                "Shareable URL (select all and copy):",
                value=st.session_state.shareable_url,
                height=100,
                help="Select all text (Ctrl+A) then copy (Ctrl+C)",
                key="shareable_url_textarea"
            )
        
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
            st.subheader("Returns Distribution and Volatility Visualizations")
            
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
            st.warning("Please load portfolio data first from the Configuration page.")
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
                help="Default: 10,000 simulations. Reduce number of simulations to hasten analyses."
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
                
                # Create simulation plot with actual path (matching original script)
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Plot percentile bands
                percentiles = simulation_results['percentiles']
                x = range(len(percentiles['50']))
                ax.fill_between(x, percentiles['5'], percentiles['95'], 
                              color='lightblue', alpha=0.3, label='5th-95th Percentile')
                ax.fill_between(x, percentiles['25'], percentiles['75'], 
                              color='blue', alpha=0.3, label='25th-75th Percentile')
                ax.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')
                
                # Plot actual path
                ax.plot(x, actual_path, 'orange', linewidth=3,
                       label=f'Actual ({actual_final_return:.2f}%, {actual_percentile:.1f}%ile)')
                
                # Format plot
                ax.set_title(f'Walk-Forward Test: {period_length} Days ({test_dates[0]} to {test_dates[-1]})', 
                           fontsize=14)
                ax.set_xlabel('Trading Days', fontsize=12)
                ax.set_ylabel('Cumulative Return (%)', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                
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
                
                # Add comprehensive drawdown analysis
                st.subheader("Comprehensive Drawdown Analysis")
                
                # Calculate actual cumulative return path for drawdown analysis
                actual_return_path = [0.0]  # Start with 0% return
                cumulative_return = 0.0
                
                for r in test_returns:
                    r_decimal = r / 100.0
                    cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
                    actual_return_path.append(cumulative_return)
                
                # Run comprehensive drawdown analysis
                drawdown_stats = analyze_drawdowns_comprehensive(
                    actual_return_path,
                    [test_dates[0]] + test_dates,  # Add initial date for 0% return point
                    period_length,
                    test_dates[0],
                    test_dates[-1],
                    data['name']
                )
                
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
            st.warning("Please load portfolio data first from the Configuration page.")
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
                                        help="Default: 10,000 simulations. Reduce number of simulations to hasten analyses.")
        
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
                        'max_drawdown': np.mean(simulation_results['max_drawdowns']), # Use mean of max_drawdowns for rolling
                        'actual_path': actual_path,
                        'simulation_results': simulation_results,
                        'test_data': test_data,
                        'train_size': len(train_data)
                    })
                
                st.session_state.analysis_results['rolling_walk'] = results
                
                # Display individual iteration plots
                if results:
                    st.subheader("Individual Iteration Analysis")
                    
                    # Add option to show/hide individual plots
                    show_individual_plots = st.checkbox("Show individual iteration plots", value=False)
                    
                    if show_individual_plots:
                        for result in results:
                            st.subheader(f"Iteration {result['iteration']} Analysis")
                            
                            # Create Monte Carlo simulation plot with actual path
                            fig_iteration, ax_iteration = plt.subplots(figsize=(12, 8))
                            
                            # Plot percentile bands
                            percentiles = result['simulation_results']['percentiles']
                            x = range(len(percentiles['50']))
                            ax_iteration.fill_between(x, percentiles['5'], percentiles['95'], 
                                                    color='lightblue', alpha=0.3, label='5th-95th Percentile')
                            ax_iteration.fill_between(x, percentiles['25'], percentiles['75'], 
                                                    color='blue', alpha=0.3, label='25th-75th Percentile')
                            ax_iteration.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')
                            
                            # Plot actual path
                            ax_iteration.plot(x, result['actual_path'], 'orange', linewidth=3,
                                            label=f'Actual ({result["actual_return"]:.2f}%, {result["percentile"]:.1f}%ile)')
                            
                            # Format plot
                            ax_iteration.set_title(f'Rolling Walk Test: Iteration {result["iteration"]} ({result["test_start_date"]} to {result["test_end_date"]})', 
                                                 fontsize=14)
                            ax_iteration.set_xlabel('Trading Days', fontsize=12)
                            ax_iteration.set_ylabel('Cumulative Return (%)', fontsize=12)
                            ax_iteration.grid(True, alpha=0.3)
                            ax_iteration.legend(loc='best')
                            
                            # Add training size information
                            ax_iteration.text(0.02, 0.02, f'Training Size: {result["train_size"]} days',
                                            transform=ax_iteration.transAxes, fontsize=10,
                                            horizontalalignment='left', verticalalignment='bottom',
                                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            st.pyplot(fig_iteration)
                            
                            # Display iteration results
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Actual Return", f"{result['actual_return']:.2f}%")
                            with col2:
                                st.metric("Forecast Return", f"{result['forecast_return']:.2f}%")
                            with col3:
                                st.metric("Percentile Rank", f"{result['percentile']:.1f}%")
                            with col4:
                                error = result['actual_return'] - result['forecast_return']
                                st.metric("Forecast Error", f"{error:.2f}%")
                            
                            # Add drawdown analysis for this iteration
                            st.subheader(f"Iteration {result['iteration']} Drawdown Analysis")
                            
                            # Run comprehensive drawdown analysis for this iteration
                            iteration_drawdown_stats = analyze_drawdowns_comprehensive(
                                result['actual_path'],
                                [result['test_start_date']] + [result['test_start_date']] * len(result['test_data']),  # Simplified dates
                                len(result['test_data']),
                                result['test_start_date'],
                                result['test_end_date'],
                                f"{data['name']}_rolling_iteration_{result['iteration']}"
                            )
                            
                            st.divider()
                
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
            st.warning("Please load portfolio data first from the Configuration page.")
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
            help="Default: 10,000 simulations. Reduce number of simulations to hasten analyses."
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
                
                # Create individual iteration plot (matching original script)
                st.subheader(f"Iteration {i + 1} Analysis")
                
                # Create Monte Carlo simulation plot with actual path
                fig_iteration, ax_iteration = plt.subplots(figsize=(12, 8))
                
                # Plot percentile bands
                percentiles = simulation_results['percentiles']
                x = range(len(percentiles['50']))
                ax_iteration.fill_between(x, percentiles['5'], percentiles['95'], 
                                        color='lightblue', alpha=0.3, label='5th-95th Percentile')
                ax_iteration.fill_between(x, percentiles['25'], percentiles['75'], 
                                        color='blue', alpha=0.3, label='25th-75th Percentile')
                ax_iteration.plot(x, percentiles['50'], 'b-', linewidth=2, label='Median Path')
                
                # Plot actual path
                ax_iteration.plot(x, actual_path, 'orange', linewidth=3,
                                label=f'Actual ({actual_final_return:.2f}%, {percentile:.1f}%ile)')
                
                # Format plot
                ax_iteration.set_title(f'Expanding Window Test: Iteration {i + 1} ({test_start_date} to {test_end_date})', 
                                     fontsize=14)
                ax_iteration.set_xlabel('Trading Days', fontsize=12)
                ax_iteration.set_ylabel('Cumulative Return (%)', fontsize=12)
                ax_iteration.grid(True, alpha=0.3)
                ax_iteration.legend(loc='best')
                
                # Add training size information
                ax_iteration.text(0.02, 0.02, f'Training Size: {train_size} days ({train_start_date} to {train_end_date})',
                                transform=ax_iteration.transAxes, fontsize=10,
                                horizontalalignment='left', verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                st.pyplot(fig_iteration)
                
                # Display iteration results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Actual Return", f"{actual_final_return:.2f}%")
                with col2:
                    st.metric("Forecast Return", f"{simulation_results['percentiles']['50'][-1]:.2f}%")
                with col3:
                    st.metric("Percentile Rank", f"{percentile:.1f}%")
                with col4:
                    error = actual_final_return - simulation_results['percentiles']['50'][-1]
                    st.metric("Forecast Error", f"{error:.2f}%")
                
                # Add drawdown analysis for this iteration
                st.subheader(f"Iteration {i + 1} Drawdown Analysis")
                
                # Run comprehensive drawdown analysis for this iteration
                iteration_drawdown_stats = analyze_drawdowns_comprehensive(
                    actual_path,
                    [test_dates[0]] + test_dates,  # Add initial date for 0% return point
                    len(test_data),
                    test_start_date,
                    test_end_date,
                    f"{data['name']}_iteration_{i + 1}"
                )
                
                # Calculate CAGR for longer periods
                if len(test_data) >= 20:  # Calculate CAGR for periods >= 20 days
                    years = len(test_data) / 252
                    actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100
                    cagr_values = [((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns]
                    cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
                    forecast_cagr = ((1 + simulation_results['percentiles']['50'][-1] / 100) ** (1 / years) - 1) * 100
                else:
                    actual_cagr = actual_final_return
                    cagr_percentile = percentile
                    forecast_cagr = simulation_results['percentiles']['50'][-1]
                
                results.append({
                    'iteration': i + 1,
                    'train_size': train_size,
                    'train_period': f"{train_start_date} to {train_end_date}",
                    'test_period': f"{test_start_date} to {test_end_date}",
                    'actual_return': actual_final_return,
                    'forecast_return': simulation_results['percentiles']['50'][-1],
                    'percentile': percentile,
                    'actual_cagr': actual_cagr,
                    'forecast_cagr': forecast_cagr,
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
                        'Forecast CAGR (%)': f"{result['forecast_cagr']:.2f}",
                        'CAGR Percentile': f"{result['cagr_percentile']:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                # Visualizations
                st.subheader("Expanding Window Analysis")
                
                # Create date labels for x-axis (convert to month/year format)
                date_labels = []
                for result in results:
                    test_start_date = result['test_period'].split(" to ")[0]  # Extract start date
                    try:
                        dt = datetime.strptime(test_start_date, '%Y-%m-%d')
                        date_labels.append(dt.strftime('%b %Y'))  # Format as "Jan 2023"
                    except:
                        # Fallback to iteration number if date conversion fails
                        date_labels.append(f"Period {len(date_labels) + 1}")
                
                # 1. Scatter plot for "Forecast Error vs Training Window Size" with date-labeled points
                fig1, ax1 = plt.subplots(figsize=(14, 7))
                
                train_sizes = [r['train_size'] for r in results]
                forecast_errors = [r['actual_return'] - r['forecast_return'] for r in results]
                abs_errors = [abs(err) for err in forecast_errors]
                
                # Create scatter plot with training size on x-axis and absolute error on y-axis
                scatter = ax1.scatter(train_sizes, abs_errors, s=80, alpha=0.7, c=train_sizes, cmap='viridis')
                
                # Fit a trend line
                if len(train_sizes) > 1:
                    z = np.polyfit(train_sizes, abs_errors, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(train_sizes), max(train_sizes), 100)
                    ax1.plot(x_range, p(x_range), 'r--', linewidth=2)
                    
                    # Add correlation text
                    corr = np.corrcoef(train_sizes, abs_errors)[0, 1]
                    ax1.text(0.05, 0.95, f'Correlation: {corr:.2f}',
                             transform=ax1.transAxes, fontsize=12,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add test period labels to points
                for i, (x, y, lbl) in enumerate(zip(train_sizes, abs_errors, date_labels)):
                    ax1.annotate(lbl, (x, y), xytext=(5, 5), textcoords='offset points')
                
                ax1.set_title('Forecast Error vs Training Window Size', fontsize=14)
                ax1.set_xlabel('Training Window Size (Trading Days)', fontsize=12)
                ax1.set_ylabel('Absolute Forecast Error (%)', fontsize=12)
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                # 2. Bar chart for "Actual vs Forecast Returns"
                fig2, ax2 = plt.subplots(figsize=(14, 7))
                
                # Set up index for bars
                indices = np.arange(len(results))
                width = 0.35
                
                actual_returns = [r['actual_return'] for r in results]
                forecast_returns = [r['forecast_return'] for r in results]
                percentiles = [r['percentile'] for r in results]
                
                # Create bar chart of actual vs forecasted returns
                ax2.bar(indices - width / 2, actual_returns, width, label='Actual Return', color='green', alpha=0.7)
                ax2.bar(indices + width / 2, forecast_returns, width, label='Forecast Return', color='blue', alpha=0.7)
                
                # Add value labels on bars
                for i, v in enumerate(actual_returns):
                    ax2.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, 
                             rotation=90 if abs(v) > 20 else 0)
                
                for i, v in enumerate(forecast_returns):
                    ax2.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, 
                             rotation=90 if abs(v) > 20 else 0)
                
                # Customize plot
                ax2.set_xlabel('Test Period Start')
                ax2.set_ylabel('Cumulative Return (%)')
                ax2.set_title(f'Expanding Window Test: Actual vs Forecast Returns - {data["name"]}')
                ax2.set_xticks(indices)
                ax2.set_xticklabels(date_labels, rotation=45)
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.legend()
                
                # Add percentile labels
                for i, pct in enumerate(percentiles):
                    ax2.text(i, max(actual_returns[i], forecast_returns[i]) + 5, f"{pct:.0f}%ile",
                             ha='center', fontsize=9, color='red')
                
                # Add training window size labels
                for i, size in enumerate(train_sizes):
                    ax2.text(i, min(actual_returns[i], forecast_returns[i]) - 5, f"{size} days",
                             ha='center', fontsize=8, color='blue')
                
                st.pyplot(fig2)
                
                # 3. Bar chart for "Actual vs Forecast CAGR" (if we have CAGR data)
                if any('actual_cagr' in r and r['actual_cagr'] != r['actual_return'] for r in results):
                    fig3, ax3 = plt.subplots(figsize=(14, 7))
                    
                    actual_cagrs = [r['actual_cagr'] for r in results if r['actual_cagr'] != r['actual_return']]
                    forecast_cagrs = [r['forecast_cagr'] for r in results if r['actual_cagr'] != r['actual_return']]
                    
                    # Create filtered indices for CAGR data
                    cagr_indices = np.arange(len(actual_cagrs))
                    cagr_date_labels = [date_labels[i] for i, r in enumerate(results) if r['actual_cagr'] != r['actual_return']]
                    
                    # Create bar chart of actual vs forecasted CAGR
                    ax3.bar(cagr_indices - width / 2, actual_cagrs, width, label='Actual CAGR', color='green', alpha=0.7)
                    ax3.bar(cagr_indices + width / 2, forecast_cagrs, width, label='Forecast CAGR', color='blue', alpha=0.7)
                    
                    # Add value labels on bars
                    for i, v in enumerate(actual_cagrs):
                        ax3.text(i - width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, 
                                 rotation=90 if abs(v) > 20 else 0)
                    
                    for i, v in enumerate(forecast_cagrs):
                        ax3.text(i + width / 2, v + 1, f"{v:.1f}%", ha='center', fontsize=9, 
                                 rotation=90 if abs(v) > 20 else 0)
                    
                    # Customize plot
                    ax3.set_xlabel('Test Period Start')
                    ax3.set_ylabel('Annualized Return (%)')
                    ax3.set_title(f'Expanding Window Test: Actual vs Forecast CAGR - {data["name"]}')
                    ax3.set_xticks(cagr_indices)
                    ax3.set_xticklabels(cagr_date_labels, rotation=45)
                    ax3.grid(True, alpha=0.3, axis='y')
                    ax3.legend()
                    
                    st.pyplot(fig3)
                
                # 4. Bar chart for "Maximum Drawdowns"
                fig4, ax4 = plt.subplots(figsize=(14, 7))
                max_drawdowns = [r['max_drawdown'] for r in results]
                
                ax4.bar(indices, max_drawdowns, color='red', alpha=0.7)
                
                # Add value labels on bars
                for i, v in enumerate(max_drawdowns):
                    ax4.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)
                
                # Customize plot
                ax4.set_xlabel('Test Period Start')
                ax4.set_ylabel('Maximum Drawdown (%)')
                ax4.set_title(f'Expanding Window Test: Maximum Drawdowns - {data["name"]}')
                ax4.set_xticks(indices)
                ax4.set_xticklabels(date_labels, rotation=45)
                ax4.grid(True, alpha=0.3, axis='y')
                
                st.pyplot(fig4)
                
                # 5. Additional analysis plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # Plot percentile ranks over time
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    iterations = [r['iteration'] for r in results]
                    cagr_percentiles = [r['cagr_percentile'] for r in results]
                    
                    ax5.plot(iterations, percentiles, 'o-', label='Return Percentile', color='green')
                    ax5.plot(iterations, cagr_percentiles, 's-', label='CAGR Percentile', color='orange')
                    ax5.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50th Percentile')
                    ax5.set_title("Percentile Ranks Over Time")
                    ax5.set_xlabel("Iteration")
                    ax5.set_ylabel("Percentile Rank")
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
                    st.pyplot(fig5)
                
                with col2:
                    # Error analysis
                    fig6, ax6 = plt.subplots(figsize=(10, 6))
                    errors = [r['actual_return'] - r['forecast_return'] for r in results]
                    ax6.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax6.axvline(x=np.mean(errors), color='red', linestyle='--', 
                               label=f'Mean Error: {np.mean(errors):.2f}%')
                    ax6.set_title("Forecast Error Distribution")
                    ax6.set_xlabel("Forecast Error (%)")
                    ax6.set_ylabel("Frequency")
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                    st.pyplot(fig6)

    elif page == "Forward Forecast":
        st.header("Forward Forecast")
        
        if st.session_state.returns_data is None:
            st.warning("Please load portfolio data first from the Configuration page.")
            return
        
        data = st.session_state.returns_data
        returns = data['returns']
        dates = data['dates']  # Add this line to get dates
        
        st.subheader("Forecast Configuration")
        
        col1, col2, col3 = st.columns(3)
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
                help="Default: 10,000 simulations. Reduce number of simulations to hasten analyses."
            )
        
        with col3:
            num_sample_paths = st.number_input(
                "Number of sample paths to display:",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Default: 200 sample paths (matching original script)"
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
            
            # Select random sample paths based on user input (matching original script)
            all_paths = simulation_results['paths']
            num_paths = all_paths.shape[0]
            num_samples = min(num_sample_paths, num_paths)  # Use user input
            sample_indices = random.sample(range(num_paths), num_samples)
            
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
            
            # Add comprehensive drawdown analysis for historical data
            st.subheader("Historical Drawdown Analysis")
            
            # Calculate cumulative returns for historical data
            historical_cumulative = [0.0]
            cumulative_return = 0.0
            
            for r in returns:
                r_decimal = r / 100.0
                cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
                historical_cumulative.append(cumulative_return)
            
            # Run comprehensive drawdown analysis on historical data
            historical_drawdown_stats = analyze_drawdowns_comprehensive(
                historical_cumulative,
                [dates[0]] + dates,  # Add initial date for 0% return point
                len(returns),
                dates[0],
                dates[-1],
                data['name']
            )
            
            st.session_state.analysis_results['forward_forecast'] = simulation_results

    elif page == "Credits":
        st.header("Credits")
        
        st.markdown("""
        This webtool is based on code originally developed and shared by **@prairie** - Huge thanks to him for sharing this analysis framework.
        """)

if __name__ == "__main__":
    main()
