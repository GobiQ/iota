Risk Management Enhancements
pythondef calculate_additional_metrics(returns, equity_curve):
    """Add more comprehensive risk metrics"""
    return {
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'calmar_ratio': annual_return / max_drawdown,
        'var_95': np.percentile(returns, 5),  # Value at Risk
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'volatility': np.std(returns) * np.sqrt(252)
    }
Data Quality & Validation
pythondef validate_data_quality(data):
    """Add data quality checks"""
    if data.isnull().sum() > len(data) * 0.05:  # More than 5% missing
        st.warning("⚠️ High percentage of missing data detected")
    
    # Check for stock splits/dividends
    daily_returns = data.pct_change()
    extreme_moves = abs(daily_returns) > 0.15  # 15% daily moves
    if extreme_moves.sum() > 0:
        st.info(f"🔍 Detected {extreme_moves.sum()} extreme price movements")
