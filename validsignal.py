#!/usr/bin/env python3
"""
Valid Signal Analysis with Composer Symphony Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import warnings
import json

# Suppress warnings
warnings.filterwarnings('ignore')

def clean_symphony_id(symphony_id: str) -> str:
    """Clean symphony ID by removing URL parts if user pasted full URL."""
    if not symphony_id or len(symphony_id.strip()) == 0:
        return ""
    
    symph_id = symphony_id.strip()
    
    # Handle full Composer URLs
    if symph_id.startswith("https://app.composer.trade/symphony/"):
        symph_id = symph_id.split("/")[-1]
    
    # Validate Symphony ID format (should be alphanumeric, typically 20+ characters)
    if len(symph_id) < 10:
        st.warning(f"⚠️ Symphony ID '{symph_id}' seems too short. Valid IDs are typically 20+ characters.")
    
    # Check for common invalid values
    invalid_values = ['details', 'help', 'about', 'login', 'signup', 'home', 'dashboard']
    if symph_id.lower() in invalid_values:
        st.error(f"❌ '{symph_id}' is not a valid Symphony ID. Please enter a real Symphony ID from Composer.")
        return ""
    
    return symph_id

def fetch_symphony_data(symphony_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch backtest data from Composer Symphony using multiple API methods."""
    
    # Clean the symphony ID
    clean_id = clean_symphony_id(symphony_id)
    if not clean_id:
        st.error("❌ Invalid Symphony ID")
        return None
    
    st.info(f"🔍 Fetching data for Symphony ID: {clean_id}")
    
    # Method 1: Try Composer Web API
    try:
        composer_url = f"https://app.composer.trade/api/symphony/{clean_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://app.composer.trade/',
            'Origin': 'https://app.composer.trade',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        st.info(f"🌐 Trying Composer API: {composer_url}")
        response = requests.get(composer_url, headers=headers, timeout=30)
        
        st.info(f"📡 Response status: {response.status_code}")
        st.info(f"📄 Response length: {len(response.text)} characters")
        
        if response.status_code == 200:
            if response.text.strip():
                # Check if response is HTML instead of JSON
                if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                    st.error("❌ Composer API returned HTML instead of JSON")
                    st.info("💡 This usually means:")
                    st.info("   - The Symphony ID is invalid")
                    st.info("   - The API requires authentication")
                    st.info("   - The API endpoint has changed")
                    st.info("📄 Response preview: <!DOCTYPE html>...")
                    # Don't return here, continue to next method
                
                try:
                    data = response.json()
                    st.success("✅ Successfully parsed JSON from Composer API")
                    return data
                except ValueError as e:
                    # Check if it's HTML after JSON parsing fails
                    if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                        st.error("❌ Composer API returned HTML instead of JSON")
                        st.info("💡 This usually means:")
                        st.info("   - The Symphony ID is invalid")
                        st.info("   - The API requires authentication")
                        st.info("   - The API endpoint has changed")
                        st.info("📄 Response preview: <!DOCTYPE html>...")
                    else:
                        st.error(f"❌ Invalid JSON response from Composer API: {str(e)}")
                        st.info(f"📄 Response preview: {response.text[:200]}...")
                    # Don't return here, continue to next method
            else:
                st.warning("⚠️ Composer API returned empty response")
        else:
            st.warning(f"⚠️ Composer API returned status {response.status_code}")
            st.info(f"📄 Response: {response.text[:200]}...")
            
    except Exception as e:
        st.error(f"❌ Composer API method failed: {str(e)}")
    
    # Method 2: Try Firestore API
    try:
        firestore_url = f"https://firestore.googleapis.com/v1/projects/leverheads-278521/databases/(default)/documents/symphony/{clean_id}"
        
        st.info(f"🌐 Trying Firestore API: {firestore_url}")
        response = requests.get(firestore_url, timeout=30)
        
        st.info(f"📡 Firestore response status: {response.status_code}")
        
        if response.status_code == 200:
            if response.text.strip():
                try:
                    data = response.json()
                    if 'fields' in data:
                        st.success("✅ Successfully parsed JSON from Firestore API")
                        return data
                    else:
                        st.warning("⚠️ Firestore response missing 'fields'")
                except ValueError as e:
                    st.error(f"❌ Invalid JSON response from Firestore API: {str(e)}")
                    st.info(f"📄 Firestore response preview: {response.text[:200]}...")
            else:
                st.warning("⚠️ Firestore API returned empty response")
        else:
            st.warning(f"⚠️ Firestore API returned status {response.status_code}")
            st.info(f"📄 Firestore response: {response.text[:200]}...")
            
    except Exception as e:
        st.error(f"❌ Firestore API method failed: {str(e)}")
    
    # Method 3: Try alternative Composer endpoint
    try:
        alt_composer_url = f"https://app.composer.trade/api/symphonies/{clean_id}"
        st.info(f"🌐 Trying alternative Composer API: {alt_composer_url}")
        
        response = requests.get(alt_composer_url, headers=headers, timeout=30)
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                st.success("✅ Successfully parsed JSON from alternative Composer API")
                return data
            except ValueError:
                st.warning("⚠️ Alternative Composer API returned invalid JSON")
        else:
            st.warning(f"⚠️ Alternative Composer API returned status {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Alternative Composer API method failed: {str(e)}")
    
    # Method 4: Try Composer API with different authentication pattern
    try:
        auth_composer_url = f"https://api.composer.trade/v1/symphonies/{clean_id}/backtest"
        st.info(f"🌐 Trying Composer API v1: {auth_composer_url}")
        
        auth_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer public'  # Try public access
        }
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'format': 'json'
        }
        
        response = requests.get(auth_composer_url, headers=auth_headers, params=params, timeout=30)
        
        st.info(f"📡 Composer API v1 response status: {response.status_code}")
        
        if response.status_code == 200 and response.text.strip():
            # Check for HTML response
            if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                st.warning("⚠️ Composer API v1 returned HTML")
            else:
                try:
                    data = response.json()
                    st.success("✅ Successfully parsed JSON from Composer API v1")
                    return data
                except ValueError:
                    st.warning("⚠️ Composer API v1 returned invalid JSON")
        else:
            st.warning(f"⚠️ Composer API v1 returned status {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Composer API v1 method failed: {str(e)}")
    
    # Method 5: Try to extract data from Composer's web interface
    try:
        web_url = f"https://app.composer.trade/symphony/{clean_id}"
        st.info(f"🌐 Trying to access Symphony web page: {web_url}")
        st.info("💡 This method attempts to extract data from the public web interface")
        
        web_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(web_url, headers=web_headers, timeout=30)
        
        if response.status_code == 200:
            st.info("✅ Successfully accessed Symphony web page")
            st.info("💡 Note: Web scraping is limited and may not provide full data")
            st.info("📄 Web page length: {} characters".format(len(response.text)))
            
            # For now, just indicate that web access worked
            st.warning("⚠️ Web interface access successful, but data extraction requires additional development")
        else:
            st.warning(f"⚠️ Web interface returned status {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Web interface method failed: {str(e)}")
    
    st.error("❌ Failed to fetch data from all available APIs")
    st.info("💡 This might mean:")
    st.info("   - The Symphony ID is incorrect")
    st.info("   - The Symphony is private or not accessible")
    st.info("   - Composer's API structure has changed")
    st.info("   - The API requires authentication or has changed")
    
    # Always offer sample data when APIs fail
    st.info("🧪 Would you like to test with sample data?")
    if st.button("📊 Create Sample Data for Testing"):
        st.info("📊 Creating sample data for demonstration...")
        return create_sample_symphony_data()
    
    return None

def create_sample_symphony_data() -> Dict[str, Any]:
    """Create sample symphony data for testing purposes."""
    import random
    from datetime import datetime, timedelta
    
    # Generate sample dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample returns
    np.random.seed(42)  # For reproducible results
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% daily return, 2% volatility
    
    # Create returns data
    returns_data = []
    for i, date in enumerate(dates):
        returns_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'return': returns[i]
        })
    
    # Create sample signals
    signals_data = []
    signal_types = ['RSI_OVERSOLD', 'MA_CROSSOVER', 'VOLUME_BREAKOUT', 'MOMENTUM_SIGNAL']
    
    for i in range(50):  # 50 sample signals
        signal_date = random.choice(dates)
        signals_data.append({
            'date': signal_date.strftime('%Y-%m-%d'),
            'signal_type': random.choice(['BUY', 'SELL']),
            'branch_name': random.choice(signal_types),
            'execution_price': round(random.uniform(100, 200), 2),
            'quantity': random.randint(1, 10)
        })
    
    return {
        'returns': returns_data,
        'signals': signals_data,
        'symphony_name': 'Sample Symphony (Demo)'
    }

def fetch_signal_data(symphony_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch detailed signal data from Composer Symphony."""
    
    # Clean the symphony ID
    clean_id = clean_symphony_id(symphony_id)
    if not clean_id:
        return None
    
    st.info(f"🔍 Fetching signal data for Symphony ID: {clean_id}")
    
    # Try to get signal data from the main symphony data
    symphony_data = fetch_symphony_data(symphony_id, start_date, end_date)
    
    if symphony_data and 'signals' in symphony_data:
        st.success("✅ Found signals in main symphony data")
        return {'signals': symphony_data['signals']}
    
    # If no signals in main data, try separate signal endpoint
    try:
        signal_url = f"https://app.composer.trade/api/symphony/{clean_id}/signals"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://app.composer.trade/',
            'Origin': 'https://app.composer.trade'
        }
        
        params = {
            'start_date': start_date,
            'end_date': end_date
        }
        
        st.info(f"🌐 Trying signal endpoint: {signal_url}")
        response = requests.get(signal_url, headers=headers, params=params, timeout=30)
        
        st.info(f"📡 Signal API response status: {response.status_code}")
        st.info(f"📄 Signal API response length: {len(response.text)} characters")
        
        if response.status_code == 200:
            if response.text.strip():
                # Check if response is HTML instead of JSON
                if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                    st.error("❌ Signal API returned HTML instead of JSON")
                    st.info("💡 This usually means the Symphony ID is invalid or the API requires authentication")
                    st.info("📄 Response preview: <!DOCTYPE html>...")
                    # Don't return here, continue to next method
                
                try:
                    data = response.json()
                    st.success("✅ Successfully parsed signal data JSON")
                    return data
                except ValueError as e:
                    # Check if it's HTML after JSON parsing fails
                    if response.text.strip().startswith('<!DOCTYPE html>') or response.text.strip().startswith('<html'):
                        st.error("❌ Signal API returned HTML instead of JSON")
                        st.info("💡 This usually means the Symphony ID is invalid or the API requires authentication")
                        st.info("📄 Response preview: <!DOCTYPE html>...")
                    else:
                        st.error(f"❌ Invalid JSON response from signal API: {str(e)}")
                        st.info(f"📄 Signal response preview: {response.text[:200]}...")
                    # Don't return here, continue to next method
            else:
                st.warning("⚠️ Signal API returned empty response")
        else:
            st.warning(f"⚠️ Signal API returned status {response.status_code}")
            st.info(f"📄 Signal response: {response.text[:200]}...")
            
    except Exception as e:
        st.error(f"❌ Signal API method failed: {str(e)}")
    
    # Try alternative signal endpoint
    try:
        alt_signal_url = f"https://app.composer.trade/api/symphonies/{clean_id}/signals"
        st.info(f"🌐 Trying alternative signal endpoint: {alt_signal_url}")
        
        response = requests.get(alt_signal_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                st.success("✅ Successfully parsed signal data from alternative endpoint")
                return data
            except ValueError:
                st.warning("⚠️ Alternative signal endpoint returned invalid JSON")
        else:
            st.warning(f"⚠️ Alternative signal endpoint returned status {response.status_code}")
            
    except Exception as e:
        st.error(f"❌ Alternative signal API method failed: {str(e)}")
    
    st.warning("⚠️ No signal data found - analysis will continue with overall performance only")
    return None

def parse_symphony_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Parse Composer Symphony data into a pandas DataFrame."""
    
    if not data:
        st.error("❌ No data received from Composer API")
        return pd.DataFrame()
    
    try:
        # Try to extract returns from different possible data structures
        returns_data = None
        
        # Method 1: Direct returns field
        if 'returns' in data:
            returns_data = data['returns']
        
        # Method 2: Backtest data
        elif 'backtest' in data and 'returns' in data['backtest']:
            returns_data = data['backtest']['returns']
        
        # Method 3: Performance data
        elif 'performance' in data and 'returns' in data['performance']:
            returns_data = data['performance']['returns']
        
        # Method 4: Firestore fields structure
        elif 'fields' in data:
            fields = data['fields']
            if 'returns' in fields:
                returns_data = fields['returns']['arrayValue']['values']
            elif 'backtest' in fields:
                backtest_field = fields['backtest']['mapValue']['fields']
                if 'returns' in backtest_field:
                    returns_data = backtest_field['returns']['arrayValue']['values']
        
        if returns_data is None:
            st.error("❌ Could not find returns data in Composer response")
            st.info(f"Available fields: {list(data.keys())}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        if isinstance(returns_data, list):
            df = pd.DataFrame(returns_data)
        else:
            # Handle different data formats
            df = pd.DataFrame([returns_data])
        
        # Ensure we have the required columns
        if 'date' in df.columns and 'return' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        elif 'timestamp' in df.columns and 'value' in df.columns:
            # Alternative column names
            df = df.rename(columns={'timestamp': 'date', 'value': 'return'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        else:
            st.error("❌ Missing required columns (date, return) in Symphony data")
            st.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error parsing Symphony data: {str(e)}")
        return pd.DataFrame()

def parse_signal_data(signal_data: Dict[str, Any]) -> pd.DataFrame:
    """Parse Composer signal data into a pandas DataFrame."""
    
    if not signal_data:
        st.error("❌ No signal data received from Composer API")
        return pd.DataFrame()
    
    try:
        # Try to extract signals from different possible data structures
        signals = None
        
        # Method 1: Direct signals field
        if 'signals' in signal_data:
            signals = signal_data['signals']
        
        # Method 2: Firestore fields structure
        elif 'fields' in signal_data:
            fields = signal_data['fields']
            if 'signals' in fields:
                signals = fields['signals']['arrayValue']['values']
        
        # Method 3: Check if signal_data is already a list
        elif isinstance(signal_data, list):
            signals = signal_data
        
        if signals is None:
            st.info("❌ Could not find signals data in Composer response")
            return pd.DataFrame()
        
        # Convert to DataFrame
        if isinstance(signals, list):
            df = pd.DataFrame(signals)
        else:
            df = pd.DataFrame([signals])
        
        # Handle different column name possibilities
        column_mapping = {
            'timestamp': 'date',
            'signal_time': 'date',
            'type': 'signal_type',
            'signal_type': 'signal_type',
            'branch': 'branch_name',
            'signal_branch': 'branch_name',
            'price': 'execution_price',
            'execution_price': 'execution_price',
            'size': 'quantity',
            'amount': 'quantity'
        }
        
        # Rename columns if needed
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure we have at least date and some signal information
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # If we don't have branch_name, create a default one
            if 'branch_name' not in df.columns:
                df['branch_name'] = 'Default Branch'
            
            # If we don't have signal_type, try to infer from other columns
            if 'signal_type' not in df.columns:
                if 'type' in df.columns:
                    df['signal_type'] = df['type']
                else:
                    df['signal_type'] = 'Unknown'
            
            return df
        else:
            st.error("❌ Missing date column in signal data")
            st.info(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error parsing signal data: {str(e)}")
        return pd.DataFrame()

def calculate_signal_metrics(signals_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate metrics for each signal branch."""
    
    if signals_df.empty:
        return {}
    
    signal_metrics = {}
    
    # Group by signal branch
    for branch_name in signals_df['branch_name'].unique():
        branch_signals = signals_df[signals_df['branch_name'] == branch_name]
        
        # Calculate metrics for this branch
        total_signals = len(branch_signals)
        
        # Calculate P&L for each signal
        branch_pnl = []
        for _, signal in branch_signals.iterrows():
            signal_date = signal['date']
            # Find corresponding return data
            if signal_date in returns_df['date'].values:
                daily_return = returns_df[returns_df['date'] == signal_date]['return'].iloc[0]
                branch_pnl.append(daily_return)
        
        if branch_pnl:
            branch_pnl = pd.Series(branch_pnl)
            
            # Calculate metrics
            win_rate = (branch_pnl > 0).mean() * 100
            avg_return = branch_pnl.mean() * 100
            total_return = branch_pnl.sum() * 100
            best_signal = branch_pnl.max() * 100
            worst_signal = branch_pnl.min() * 100
            
            signal_metrics[branch_name] = {
                'Total Signals': total_signals,
                'Win Rate': win_rate,
                'Average Return': avg_return,
                'Total Return': total_return,
                'Best Signal': best_signal,
                'Worst Signal': worst_signal,
                'Success Count': (branch_pnl > 0).sum(),
                'Failure Count': (branch_pnl <= 0).sum()
            }
    
    return signal_metrics

def calculate_backtest_metrics(returns_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive backtest metrics from returns data."""
    
    if returns_df.empty:
        return {
            'Total Days': 0,
            'Total Return': 0.0,
            'Annualized Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Volatility': 0.0,
            'Win Rate': 0.0,
            'Best Day': 0.0,
            'Worst Day': 0.0
        }
    
    # Convert returns to numeric
    returns = pd.to_numeric(returns_df['return'], errors='coerce').dropna()
    
    if returns.empty:
        return {
            'Total Days': len(returns_df),
            'Total Return': 0.0,
            'Annualized Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Max Drawdown': 0.0,
            'Volatility': 0.0,
            'Win Rate': 0.0,
            'Best Day': 0.0,
            'Worst Day': 0.0
        }
    
    # Calculate metrics
    total_days = len(returns)
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) if total_days > 1 else 0
    
    # Sharpe Ratio
    risk_free_rate = 0.02  # Assuming 2% risk-free rate
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win Rate
    win_rate = (returns > 0).mean() * 100
    
    # Best and Worst Days
    best_day = returns.max()
    worst_day = returns.min()
    
    return {
        'Total Days': total_days,
        'Total Return': total_return * 100,
        'Annualized Return': annualized_return * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown * 100,
        'Volatility': volatility * 100,
        'Win Rate': win_rate,
        'Best Day': best_day * 100,
        'Worst Day': worst_day * 100
    }

def main():
    """Main function for the Composer Symphony analysis app."""
    
    st.title("🔍 Composer Symphony Analysis")
    st.markdown("Analyze backtest performance and trading characteristics from Composer Symphony data.")
    
    # Input section
    st.subheader("📊 Symphony Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symphony_id = st.text_input(
            "Symphony ID or URL",
            placeholder="Enter Symphony ID or full Composer URL",
            help="Enter the Symphony ID (e.g., GiR9AkRAZ1S4IONmYIkS) or full Composer URL"
        )
        
        # Show example
        with st.expander("💡 How to find your Symphony ID"):
            st.markdown("""
            **Option 1: From Composer URL**
            - Go to your Symphony on Composer
            - Copy the URL: `https://app.composer.trade/symphony/GiR9AkRAZ1S4IONmYIkS`
            - Paste the full URL or just the ID: `GiR9AkRAZ1S4IONmYIkS`
            
            **Option 2: From Symphony Page**
            - Open your Symphony in Composer
            - Look at the URL in your browser
            - The ID is the last part after `/symphony/`
            
            **Valid Symphony ID Examples:**
            - `GiR9AkRAZ1S4IONmYIkS` (20+ characters, alphanumeric)
            - `abc123def456ghi789jkl` (typical format)
            
            **Invalid Examples:**
            - `details` ❌ (too short, not a real ID)
            - `help` ❌ (not a Symphony ID)
            - `123` ❌ (too short)
            """)
            
            st.markdown("""
            **💡 Tip:** If you don't have a Symphony ID, you can:
            1. Create a new Symphony on Composer
            2. Use the sample data option for testing
            3. Ask someone with a Symphony to share their ID
            """)
    
    with col2:
        # Date range selection
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            help="Select the date range for analysis"
        )
    
    # Analysis parameters
    st.subheader("⚙️ Analysis Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Risk-free rate for Sharpe/Sortino calculations"
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            options=[0.90, 0.95, 0.99],
            index=1,
            help="Confidence level for statistical analysis"
        )
    
    with col3:
        min_data_days = st.number_input(
            "Minimum Data Days",
            min_value=30,
            max_value=1000,
            value=90,
            help="Minimum number of days required for analysis"
        )
    
    # Run analysis button
    if st.button("🚀 Analyze Symphony", type="primary"):
        if not symphony_id:
            st.error("❌ Please enter a Symphony ID")
            return
        
        # Validate Symphony ID before making API calls
        clean_id = clean_symphony_id(symphony_id)
        if not clean_id:
            st.error("❌ Please enter a valid Symphony ID")
            st.info("💡 Valid Symphony IDs are typically 20+ characters long and alphanumeric")
            return
        
        if len(date_range) != 2:
            st.error("❌ Please select a valid date range")
            return
        
        start_date, end_date = date_range
        
        with st.spinner("🔄 Fetching Symphony data..."):
            # Fetch both backtest and signal data from Composer
            backtest_data = fetch_symphony_data(
                symphony_id,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            signal_data = fetch_signal_data(
                symphony_id,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        
        if backtest_data is None:
            st.error("❌ Failed to fetch backtest data from Composer")
            return
        
        # Parse the data
        returns_df = parse_symphony_data(backtest_data)
        signals_df = parse_signal_data(signal_data) if signal_data else pd.DataFrame()
        
        if returns_df.empty:
            st.error("❌ No valid data found for the specified period")
            return
        
        if len(returns_df) < min_data_days:
            st.warning(f"⚠️ Limited data: Only {len(returns_df)} days available (minimum: {min_data_days})")
        
        # Calculate overall metrics
        metrics = calculate_backtest_metrics(returns_df)
        
        # Calculate signal-level metrics if signal data is available
        signal_metrics = {}
        if not signals_df.empty:
            signal_metrics = calculate_signal_metrics(signals_df, returns_df)
        
        # Display results
        st.success("✅ Analysis completed successfully!")
        
        # Overall Performance metrics
        st.subheader("📈 Overall Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{metrics['Total Return']:.2f}%")
            st.metric("Annualized Return", f"{metrics['Annualized Return']:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        
        with col2:
            st.metric("Sortino Ratio", f"{metrics['Sortino Ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")
            st.metric("Volatility", f"{metrics['Volatility']:.2f}%")
        
        with col3:
            st.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
            st.metric("Best Day", f"{metrics['Best Day']:.2f}%")
            st.metric("Worst Day", f"{metrics['Worst Day']:.2f}%")
        
        # Signal Branch Analysis
        if signal_metrics:
            st.subheader("🎯 Signal Branch Analysis")
            st.markdown("Breakdown of performance by individual signal branches:")
            
            # Create signal metrics table
            signal_data = []
            for branch_name, branch_metrics in signal_metrics.items():
                signal_data.append({
                    'Branch Name': branch_name,
                    'Total Signals': branch_metrics['Total Signals'],
                    'Win Rate (%)': f"{branch_metrics['Win Rate']:.1f}",
                    'Avg Return (%)': f"{branch_metrics['Average Return']:.2f}",
                    'Total Return (%)': f"{branch_metrics['Total Return']:.2f}",
                    'Success/Failure': f"{branch_metrics['Success Count']}/{branch_metrics['Failure Count']}",
                    'Best Signal (%)': f"{branch_metrics['Best Signal']:.2f}",
                    'Worst Signal (%)': f"{branch_metrics['Worst Signal']:.2f}"
                })
            
            signal_df = pd.DataFrame(signal_data)
            st.dataframe(signal_df, use_container_width=True)
            
            # Signal branch performance visualization
            st.subheader("📊 Signal Branch Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Win rate by branch
                branch_names = list(signal_metrics.keys())
                win_rates = [signal_metrics[branch]['Win Rate'] for branch in branch_names]
                
                fig_win_rate = go.Figure(data=[go.Bar(x=branch_names, y=win_rates)])
                fig_win_rate.update_layout(
                    title="Win Rate by Signal Branch",
                    xaxis_title="Signal Branch",
                    yaxis_title="Win Rate (%)",
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig_win_rate, use_container_width=True)
            
            with col2:
                # Total return by branch
                total_returns = [signal_metrics[branch]['Total Return'] for branch in branch_names]
                
                fig_total_return = go.Figure(data=[go.Bar(x=branch_names, y=total_returns)])
                fig_total_return.update_layout(
                    title="Total Return by Signal Branch",
                    xaxis_title="Signal Branch",
                    yaxis_title="Total Return (%)"
                )
                st.plotly_chart(fig_total_return, use_container_width=True)
            
            # Signal frequency analysis
            st.subheader("📈 Signal Frequency Analysis")
            
            if not signals_df.empty:
                # Signal frequency over time
                signal_counts = signals_df.groupby(['date', 'branch_name']).size().reset_index(name='count')
                
                fig_frequency = go.Figure()
                for branch in signals_df['branch_name'].unique():
                    branch_data = signal_counts[signal_counts['branch_name'] == branch]
                    fig_frequency.add_trace(go.Scatter(
                        x=branch_data['date'],
                        y=branch_data['count'],
                        mode='lines+markers',
                        name=branch
                    ))
                
                fig_frequency.update_layout(
                    title="Signal Frequency Over Time",
                    xaxis_title="Date",
                    yaxis_title="Number of Signals"
                )
                st.plotly_chart(fig_frequency, use_container_width=True)
        
        # Data summary
        st.subheader("📋 Data Summary")
        st.info(f"**Analysis Period**: {start_date} to {end_date} ({metrics['Total Days']} trading days)")
        if signal_metrics:
            st.info(f"**Total Signal Branches**: {len(signal_metrics)}")
            total_signals = sum(signal_metrics[branch]['Total Signals'] for branch in signal_metrics)
            st.info(f"**Total Signals Fired**: {total_signals}")
        
        # Returns distribution
        st.subheader("📊 Returns Distribution")
        
        if not returns_df.empty:
            returns = pd.to_numeric(returns_df['return'], errors='coerce').dropna()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = go.Figure(data=[go.Histogram(x=returns*100, nbinsx=30)])
                fig_hist.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Cumulative returns
                cumulative_returns = (1 + returns).cumprod()
                fig_cum = go.Figure(data=[go.Scatter(x=returns_df['date'], y=cumulative_returns)])
                fig_cum.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return"
                )
                st.plotly_chart(fig_cum, use_container_width=True)
        
        # Performance interpretation
        st.subheader("🎯 Performance Interpretation")
        
        # Risk assessment
        risk_level = "LOW" if metrics['Max Drawdown'] < 10 else "MEDIUM" if metrics['Max Drawdown'] < 20 else "HIGH"
        performance_rating = "EXCELLENT" if metrics['Sharpe Ratio'] > 1.5 else "GOOD" if metrics['Sharpe Ratio'] > 1.0 else "POOR"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Risk Level**: {risk_level} (Max Drawdown: {metrics['Max Drawdown']:.2f}%)")
            st.info(f"**Performance Rating**: {performance_rating} (Sharpe: {metrics['Sharpe Ratio']:.2f})")
        
        with col2:
            st.info(f"**Consistency**: {'HIGH' if metrics['Win Rate'] > 60 else 'MEDIUM' if metrics['Win Rate'] > 50 else 'LOW'} ({metrics['Win Rate']:.1f}% win rate)")
            st.info(f"**Volatility**: {'LOW' if metrics['Volatility'] < 15 else 'MEDIUM' if metrics['Volatility'] < 25 else 'HIGH'} ({metrics['Volatility']:.2f}%)")
        
        # Signal branch insights
        if signal_metrics:
            st.subheader("💡 Signal Branch Insights")
            
            # Find best and worst performing branches
            best_branch = max(signal_metrics.items(), key=lambda x: x[1]['Win Rate'])
            worst_branch = min(signal_metrics.items(), key=lambda x: x[1]['Win Rate'])
            most_active_branch = max(signal_metrics.items(), key=lambda x: x[1]['Total Signals'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"**Best Performing Branch**: {best_branch[0]} ({best_branch[1]['Win Rate']:.1f}% win rate)")
            
            with col2:
                st.warning(f"**Worst Performing Branch**: {worst_branch[0]} ({worst_branch[1]['Win Rate']:.1f}% win rate)")
            
            with col3:
                st.info(f"**Most Active Branch**: {most_active_branch[0]} ({most_active_branch[1]['Total Signals']} signals)")

if __name__ == "__main__":
    main() 
