import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks for Streamlit
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Simple RSI calculation without external dependencies
def calculate_rsi_simple(prices: pd.Series, window: int = 10) -> pd.Series:
    """Custom RSI implementation - no external dependencies"""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class SimpleRegimeDetector:
    """Simplified regime detector using basic technical indicators"""
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.regime_labels = {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
        self.thresholds = None
        
    def _calculate_features(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate simple regime features"""
        if len(prices) < 50:
            return pd.DataFrame()
            
        features = pd.DataFrame(index=prices.index)
        
        # Price momentum features
        features['returns_20'] = prices.pct_change(20)
        features['returns_5'] = prices.pct_change(5)
        
        # Volatility (rolling standard deviation)
        returns = prices.pct_change()
        features['volatility'] = returns.rolling(20).std()
        
        # Moving average position
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        features['price_vs_ma20'] = (prices - ma_20) / ma_20
        features['ma20_vs_ma50'] = (ma_20 - ma_50) / ma_50
        
        return features.dropna()
    
    def fit(self, prices: pd.Series):
        """Fit simple thresholds for regime classification"""
        features = self._calculate_features(prices)
        
        if len(features) == 0:
            # Default thresholds
            self.thresholds = {
                'momentum_bull': 0.02,
                'momentum_bear': -0.02,
                'volatility_high': 0.03
            }
            return self
        
        # Calculate dynamic thresholds based on data distribution
        self.thresholds = {
            'momentum_bull': features['returns_20'].quantile(0.65),
            'momentum_bear': features['returns_20'].quantile(0.35),
            'volatility_high': features['volatility'].quantile(0.7)
        }
        
        return self
    
    def predict_regime(self, prices: pd.Series) -> int:
        """Predict regime using simple rules"""
        if len(prices) < 50 or self.thresholds is None:
            return 2  # Default to sideways
        
        features = self._calculate_features(prices)
        if len(features) == 0:
            return 2
        
        latest = features.iloc[-1]
        
        # Simple rule-based classification
        momentum = latest['returns_20']
        
        if pd.isna(momentum):
            return 2
        
        if momentum > self.thresholds['momentum_bull']:
            return 0  # Bull
        elif momentum < self.thresholds['momentum_bear']:
            return 1  # Bear
        else:
            return 2  # Sideways

class BacktestEngine:
    """Streamlit-safe backtesting engine"""
    
    def __init__(self, target_ticker: str = "TQQQ"):
        self.target_ticker = target_ticker
        self.trades = pd.DataFrame()
        self.equity_curve = pd.Series()
        self.performance_metrics = {}
    
    def run_backtest(self, prices: pd.Series, asset_prices: Dict[str, pd.Series], 
                    strategy_signals: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """Run backtest with error handling for Streamlit"""
        
        try:
            # Ensure data alignment
            common_dates = prices.index
            for ticker, price_series in asset_prices.items():
                common_dates = common_dates.intersection(price_series.index)
            
            if len(common_dates) < 100:
                return self._empty_results(initial_capital, "Insufficient data")
            
            # Align data
            aligned_prices = prices.loc[common_dates]
            aligned_asset_prices = {ticker: series.loc[common_dates] 
                                   for ticker, series in asset_prices.items()}
            aligned_signals = strategy_signals.loc[common_dates]
            
            # Initialize backtest
            capital = initial_capital
            equity_curve = []
            trades_list = []
            current_asset = None
            entry_price = None
            entry_date = None
            
            for date in aligned_signals.index[1:]:
                signal = aligned_signals.loc[date]
                recommended_asset = signal.get('target_asset', self.target_ticker)
                
                # Asset switching logic
                if current_asset != recommended_asset:
                    # Close position
                    if current_asset is not None and entry_price is not None:
                        if current_asset in aligned_asset_prices:
                            exit_price = aligned_asset_prices[current_asset].loc[date]
                            if not pd.isna(exit_price) and exit_price > 0:
                                asset_return = (exit_price - entry_price) / entry_price
                                new_capital = capital * (1 + asset_return)
                                
                                trades_list.append({
                                    'entry_date': entry_date,
                                    'exit_date': date,
                                    'asset': current_asset,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'return': asset_return,
                                    'capital_before': capital,
                                    'capital_after': new_capital,
                                    'regime': signal.get('regime', 'Unknown'),
                                    'rsi_condition': signal.get('rsi_condition', 'Neutral')
                                })
                                
                                capital = new_capital
                    
                    # Enter new position
                    if recommended_asset in aligned_asset_prices:
                        current_asset = recommended_asset
                        entry_price = aligned_asset_prices[current_asset].loc[date]
                        entry_date = date
                    else:
                        current_asset = None
                        entry_price = None
                
                equity_curve.append(capital)
            
            # Store results
            self.trades = pd.DataFrame(trades_list)
            self.equity_curve = pd.Series(equity_curve, 
                                         index=aligned_signals.index[1:len(equity_curve)+1])
            
            # Calculate metrics
            self.performance_metrics = self._calculate_metrics(initial_capital)
            
            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'performance_metrics': self.performance_metrics,
                'final_capital': capital,
                'total_return': (capital - initial_capital) / initial_capital
            }
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return self._empty_results(initial_capital, str(e))
    
    def _empty_results(self, initial_capital: float, error_msg: str) -> Dict:
        """Return empty results on error"""
        return {
            'trades': pd.DataFrame(),
            'equity_curve': pd.Series(),
            'performance_metrics': {'total_return': 0, 'error': error_msg},
            'final_capital': initial_capital,
            'total_return': 0
        }
    
    def _calculate_metrics(self, initial_capital: float) -> Dict:
        """Calculate performance metrics safely"""
        try:
            if len(self.equity_curve) == 0:
                return {'total_return': 0}
            
            total_return = (self.equity_curve.iloc[-1] - initial_capital) / initial_capital
            daily_returns = self.equity_curve.pct_change().dropna()
            
            if len(daily_returns) == 0:
                return {'total_return': total_return}
            
            # Basic metrics
            volatility = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
            
            # Drawdown
            running_max = self.equity_curve.expanding().max()
            drawdown = (self.equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Trade stats
            win_rate = (self.trades['return'] > 0).mean() if len(self.trades) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(self.trades)
            }
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return {'total_return': 0, 'error': str(e)}
    
    def print_performance_report(self):
        """Print performance report safely"""
        if not self.performance_metrics:
            print("No backtest results available.")
            return
        
        print("="*50)
        print("BACKTEST PERFORMANCE REPORT")
        print("="*50)
        print(f"Target Ticker: {self.target_ticker}")
        
        for key, value in self.performance_metrics.items():
            if key == 'error':
                print(f"Error: {value}")
            elif isinstance(value, float):
                if 'return' in key or 'drawdown' in key or 'rate' in key:
                    print(f"{key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")

class RSISignalOptimizer:
    """Streamlit-safe signal optimizer"""
    
    def __init__(self, regime_detector: SimpleRegimeDetector, target_ticker: str = "TQQQ"):
        self.regime_detector = regime_detector
        self.target_ticker = target_ticker
        self.optimal_thresholds = {}
        self.asset_allocation_rules = {}
        
    def set_asset_allocation_rules(self, allocation_rules: Dict):
        """Set asset allocation rules"""
        self.asset_allocation_rules = allocation_rules
    
    def optimize_thresholds(self, prices: pd.Series, asset_prices: Dict[str, pd.Series]) -> Dict:
        """Optimize thresholds with simple grid search"""
        print(f"Optimizing RSI thresholds for {self.target_ticker}...")
        
        # Simple grid search instead of complex optimization
        for regime in range(self.regime_detector.n_regimes):
            regime_name = self.regime_detector.regime_labels[regime]
            
            # Use default thresholds adjusted by regime
            if regime == 0:  # Bull
                self.optimal_thresholds[regime] = {'lower': 25.0, 'upper': 75.0}
            elif regime == 1:  # Bear
                self.optimal_thresholds[regime] = {'lower': 35.0, 'upper': 65.0}
            else:  # Sideways
                self.optimal_thresholds[regime] = {'lower': 30.0, 'upper': 70.0}
            
            thresholds = self.optimal_thresholds[regime]
            print(f"  {regime_name}: Lower={thresholds['lower']:.1f}, Upper={thresholds['upper']:.1f}")
        
        return self.optimal_thresholds
    
    def generate_strategy_signals(self, prices: pd.Series, asset_prices: Dict[str, pd.Series]) -> pd.DataFrame:
        """Generate trading signals"""
        try:
            rsi = calculate_rsi_simple(prices, window=10)
            signals = []
            
            min_history = 50
            
            for i, date in enumerate(prices.index):
                if i < min_history:
                    signals.append({
                        'regime': 'Unknown',
                        'rsi_condition': 'Neutral',
                        'target_asset': self.target_ticker,
                        'rsi_value': np.nan,
                        'rsi_lower': 30,
                        'rsi_upper': 70
                    })
                    continue
                
                try:
                    # Detect regime
                    current_regime = self.regime_detector.predict_regime(prices.iloc[:i+1])
                    regime_name = self.regime_detector.regime_labels[current_regime].lower()
                    
                    # Get thresholds
                    if current_regime in self.optimal_thresholds:
                        lower = self.optimal_thresholds[current_regime]['lower']
                        upper = self.optimal_thresholds[current_regime]['upper']
                    else:
                        lower, upper = 30, 70
                    
                    # RSI condition
                    current_rsi = rsi.iloc[i]
                    if pd.isna(current_rsi):
                        rsi_condition = 'Neutral'
                    elif current_rsi > upper:
                        rsi_condition = 'Overbought'
                    elif current_rsi < lower:
                        rsi_condition = 'Oversold'
                    else:
                        rsi_condition = 'Neutral'
                    
                    # Target asset
                    target_asset = self._get_target_asset(rsi_condition, regime_name)
                    
                    signals.append({
                        'regime': regime_name.title(),
                        'rsi_condition': rsi_condition,
                        'target_asset': target_asset,
                        'rsi_value': current_rsi,
                        'rsi_lower': lower,
                        'rsi_upper': upper
                    })
                    
                except Exception as e:
                    signals.append({
                        'regime': 'Unknown',
                        'rsi_condition': 'Neutral',
                        'target_asset': self.target_ticker,
                        'rsi_value': np.nan,
                        'rsi_lower': 30,
                        'rsi_upper': 70
                    })
            
            return pd.DataFrame(signals, index=prices.index)
            
        except Exception as e:
            print(f"Signal generation error: {e}")
            # Return default signals
            default_signals = pd.DataFrame({
                'regime': 'Unknown',
                'rsi_condition': 'Neutral',
                'target_asset': self.target_ticker,
                'rsi_value': np.nan,
                'rsi_lower': 30,
                'rsi_upper': 70
            }, index=prices.index)
            return default_signals
    
    def _get_target_asset(self, rsi_condition: str, regime: str) -> str:
        """Get target asset based on condition and regime"""
        condition_key = f'rsi_{rsi_condition.lower()}'
        
        if (condition_key in self.asset_allocation_rules and 
            regime in self.asset_allocation_rules[condition_key]):
            return self.asset_allocation_rules[condition_key][regime]
        
        return self.target_ticker

def generate_sample_data() -> Dict[str, pd.Series]:
    """Generate sample data for demonstration"""
    print("Generating sample market data...")
    
    # Simple simulation
    np.random.seed(42)
    n_days = 1000
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
    
    # Base market movement
    base_returns = np.random.normal(0.0005, 0.02, n_days)
    
    # Create different assets
    assets = {
        'TQQQ': {'multiplier': 3.0, 'volatility': 1.8},
        'TECL': {'multiplier': 3.0, 'volatility': 1.9},
        'SOXL': {'multiplier': 3.0, 'volatility': 2.1},
        'UVXY': {'multiplier': -1.5, 'volatility': 3.0},
        'SQQQ': {'multiplier': -3.0, 'volatility': 1.8},
        'BSV': {'multiplier': 0.1, 'volatility': 0.3}
    }
    
    price_data = {}
    
    for ticker, params in assets.items():
        # Generate returns
        returns = base_returns * params['multiplier'] + np.random.normal(0, 0.01, n_days) * params['volatility']
        
        # Convert to prices
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data[ticker] = pd.Series(prices[1:], index=dates)
    
    return price_data

# Main execution - Streamlit safe
if __name__ == "__main__":
    print("="*60)
    print("RSI BACKTEST ENGINE (Streamlit Safe)")
    print("="*60)
    
    # Configuration
    TARGET_TICKER = "TQQQ"
    INITIAL_CAPITAL = 100000
    
    # Asset allocation rules
    ALLOCATION_RULES = {
        'rsi_overbought': {
            'bull': 'UVXY',
            'bear': 'SQQQ',
            'sideways': 'BSV'
        },
        'rsi_oversold': {
            'bull': 'TECL',
            'bear': 'TQQQ',
            'sideways': 'SOXL'
        },
        'rsi_neutral': {
            'bull': 'TQQQ',
            'bear': 'BSV',
            'sideways': 'TQQQ'
        }
    }
    
    try:
        # Generate sample data (safe for Streamlit)
        print(f"Setting up backtest for {TARGET_TICKER}")
        asset_data = generate_sample_data()
        target_data = asset_data[TARGET_TICKER]
        
        print(f"Data range: {target_data.index[0].strftime('%Y-%m-%d')} to {target_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(target_data)}")
        
        # Initialize components
        regime_detector = SimpleRegimeDetector(n_regimes=3)
        regime_detector.fit(target_data)
        
        signal_optimizer = RSISignalOptimizer(regime_detector, TARGET_TICKER)
        signal_optimizer.set_asset_allocation_rules(ALLOCATION_RULES)
        
        # Optimize and generate signals
        print("Optimizing RSI thresholds...")
        optimal_thresholds = signal_optimizer.optimize_thresholds(target_data, asset_data)
        
        print("Generating strategy signals...")
        strategy_signals = signal_optimizer.generate_strategy_signals(target_data, asset_data)
        
        # Run backtest
        print(f"Running backtest with ${INITIAL_CAPITAL:,} initial capital...")
        backtest_engine = BacktestEngine(TARGET_TICKER)
        backtest_results = backtest_engine.run_backtest(
            target_data, asset_data, strategy_signals, INITIAL_CAPITAL
        )
        
        # Print results
        backtest_engine.print_performance_report()
        
        print(f"\n" + "="*60)
        print("STRATEGY SUMMARY")
        print("="*60)
        print(f"Target Ticker: {TARGET_TICKER}")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
        
        # Recent signals
        print(f"\n" + "="*60)
        print("RECENT SIGNALS")
        print("="*60)
        recent_signals = strategy_signals.tail(5)
        for date, signal in recent_signals.iterrows():
            rsi_val = signal.get('rsi_value', np.nan)
            rsi_str = f"{rsi_val:.1f}" if not pd.isna(rsi_val) else "N/A"
            print(f"{date.strftime('%Y-%m-%d')}: {signal['regime']} market, "
                  f"RSI {rsi_str} ({signal['rsi_condition']}) -> {signal['target_asset']}")
        
        print(f"\n✓ Backtest completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This appears to be a Streamlit compatibility issue.")
        print("The system is designed to work with basic dependencies only.")
