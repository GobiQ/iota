#!/usr/bin/env python3
"""
Valid Signal Analysis with Robust Error Handling
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_metrics(trades_df):
    """Calculate trading metrics from trades dataframe with robust error handling."""
    
    # Handle empty dataframe
    if trades_df.empty:
        return {
            'Total Trades': 0,
            'Win Rate': 0.0,
            'Average Return': 0.0,
            'Total Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }
    
    # Find the returns column - try multiple possible names
    returns_col = None
    possible_columns = ['return', 'returns', 'Return', 'Returns', 'profit', 'Profit', 'PnL', 'pnl', 'P&L']
    
    for col in possible_columns:
        if col in trades_df.columns:
            returns_col = col
            break
    
    if returns_col is None:
        # If no returns column found, return default values
        return {
            'Total Trades': len(trades_df),
            'Win Rate': 0.0,
            'Average Return': 0.0,
            'Total Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }
    
    # Convert to numeric, handling any non-numeric values
    try:
        returns = pd.to_numeric(trades_df[returns_col], errors='coerce')
    except Exception:
        returns = pd.Series([0.0] * len(trades_df))
    
    # Remove NaN values
    returns = returns.dropna()
    
    # Handle empty returns after cleaning
    if returns.empty:
        return {
            'Total Trades': len(trades_df),
            'Win Rate': 0.0,
            'Average Return': 0.0,
            'Total Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }
    
    # Ensure returns is numeric
    if not pd.api.types.is_numeric_dtype(returns):
        return {
            'Total Trades': len(trades_df),
            'Win Rate': 0.0,
            'Average Return': 0.0,
            'Total Return': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }
    
    # Calculate metrics with comprehensive error handling
    try:
        # Win rate calculation
        positive_returns = returns > 0
        win_rate = positive_returns.mean() * 100 if len(returns) > 0 else 0.0
    except Exception:
        win_rate = 0.0
    
    try:
        # Average return
        avg_return = returns.mean() if len(returns) > 0 else 0.0
    except Exception:
        avg_return = 0.0
    
    try:
        # Total return
        total_return = returns.sum() if len(returns) > 0 else 0.0
    except Exception:
        total_return = 0.0
    
    try:
        # Sharpe ratio
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
    except Exception:
        sharpe_ratio = 0.0
    
    try:
        # Max drawdown
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100 if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0
    except Exception:
        max_drawdown = 0.0
    
    return {
        'Total Trades': len(trades_df),
        'Win Rate': win_rate,
        'Average Return': avg_return,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def analyze_trading_signals(trades_data):
    """Analyze trading signals and calculate performance metrics."""
    
    if not trades_data or trades_data.empty:
        st.warning("No trading data provided for analysis.")
        return None
    
    # Calculate metrics
    metrics = calculate_metrics(trades_data)
    
    # Display results
    st.subheader("📊 Trading Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", metrics['Total Trades'])
        st.metric("Win Rate", f"{metrics['Win Rate']:.1f}%")
    
    with col2:
        st.metric("Average Return", f"{metrics['Average Return']:.2f}%")
        st.metric("Total Return", f"{metrics['Total Return']:.2f}%")
    
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2f}%")
    
    return metrics

def main():
    """Main function for the valid signal analysis app."""
    
    st.title("🔍 Valid Signal Analysis")
    st.markdown("Analyze trading signals and calculate performance metrics.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your trading data (CSV format)",
        type=['csv'],
        help="Upload a CSV file with trading data including returns/profit columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            trades_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(trades_df)} trading records")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(trades_df.head())
            
            # Analyze signals
            metrics = analyze_trading_signals(trades_df)
            
            if metrics:
                st.success("✅ Analysis completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file contains valid trading data with a returns/profit column.")

if __name__ == "__main__":
    main() 
