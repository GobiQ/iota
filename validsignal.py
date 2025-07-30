import numpy as np
import pandas as pd

print("RSI Backtest Engine Loading...")

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
    print("Generating test data...")
    prices = generate_test_data()
    
    print("Calculating RSI...")
    rsi = calculate_rsi_simple(prices)
    
    # Simple strategy: buy when RSI < 30, sell when RSI > 70
    signals = pd.Series(0, index=prices.index)
    signals[rsi < 30] = 1   # Buy
    signals[rsi > 70] = -1  # Sell
    
    # Calculate returns
    returns = prices.pct_change()
    strategy_returns = signals.shift(1) * returns
    
    total_return = strategy_returns.sum()
    
    print(f"\nSimple RSI Strategy Results:")
    print(f"Data period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(prices)}")
    print(f"Final RSI: {rsi.iloc[-1]:.1f}")
    print(f"Strategy return: {total_return:.2%}")
    print(f"Buy signals: {(signals == 1).sum()}")
    print(f"Sell signals: {(signals == -1).sum()}")
    
    return {
        'prices': prices,
        'rsi': rsi,
        'signals': signals,
        'total_return': total_return
    }

# Test execution
try:
    print("Starting simple backtest...")
    results = simple_backtest()
    print("✓ Backtest completed successfully!")
except Exception as e:
    print(f"Error: {e}")

print("RSI Backtest Engine loaded successfully.")
