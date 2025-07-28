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

def fetch_symphony_data(symphony_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """Fetch backtest data from Composer Symphony API."""
    
    # Composer API endpoint (you may need to adjust this based on actual API)
    api_url = f"https://api.composer.trade/v1/symphonies/{symphony_id}/backtest"
    
    params = {
        'start_date': start_date,
        'end_date': end_date,
        'format': 'json'
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching data from Composer: {str(e)}")
        return None

def parse_symphony_data(data: Dict[str, Any]) -> pd.DataFrame:
    """Parse Composer Symphony data into a pandas DataFrame."""
    
    if not data or 'returns' not in data:
        st.error("❌ Invalid data format from Composer API")
        return pd.DataFrame()
    
    try:
        # Extract returns data
        returns_data = data['returns']
        
        # Convert to DataFrame
        df = pd.DataFrame(returns_data)
        
        # Ensure we have the required columns
        if 'date' in df.columns and 'return' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            return df
        else:
            st.error("❌ Missing required columns (date, return) in Symphony data")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error parsing Symphony data: {str(e)}")
        return pd.DataFrame()

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
            "Symphony ID",
            placeholder="Enter your Composer Symphony ID",
            help="The unique identifier for your Composer Symphony"
        )
    
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
        
        if len(date_range) != 2:
            st.error("❌ Please select a valid date range")
            return
        
        start_date, end_date = date_range
        
        with st.spinner("🔄 Fetching Symphony data..."):
            # Fetch data from Composer
            data = fetch_symphony_data(
                symphony_id,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
        
        if data is None:
            st.error("❌ Failed to fetch data from Composer")
            return
        
        # Parse the data
        returns_df = parse_symphony_data(data)
        
        if returns_df.empty:
            st.error("❌ No valid data found for the specified period")
            return
        
        if len(returns_df) < min_data_days:
            st.warning(f"⚠️ Limited data: Only {len(returns_df)} days available (minimum: {min_data_days})")
        
        # Calculate metrics
        metrics = calculate_backtest_metrics(returns_df)
        
        # Display results
        st.success("✅ Analysis completed successfully!")
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
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
        
        # Data summary
        st.subheader("📋 Data Summary")
        st.info(f"**Analysis Period**: {start_date} to {end_date} ({metrics['Total Days']} trading days)")
        
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

if __name__ == "__main__":
    main() 
