import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import talib
from typing import Dict, Tuple, List, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """Detects market regimes using multiple indicators"""
    
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.scaler = StandardScaler()
        self.regime_labels = {0: 'Bull', 1: 'Bear', 2: 'Sideways'}
        
    def extract_features(self, prices: pd.Series) -> pd.DataFrame:
        """Extract regime-detection features from price data"""
        features = pd.DataFrame(index=prices.index)
        
        # Volatility features
        returns = prices.pct_change()
        features['volatility_20'] = returns.rolling(20).std()
        features['volatility_5'] = returns.rolling(5).std()
        
        # Trend features
        features['ma_slope_20'] = prices.rolling(20).mean().pct_change(5)
        features['ma_slope_50'] = prices.rolling(50).mean().pct_change(10)
        
        # Momentum features
        features['roc_10'] = (prices / prices.shift(10) - 1)
        features['roc_20'] = (prices / prices.shift(20) - 1)
        
        # Price position relative to moving averages
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        features['price_vs_ma20'] = (prices - ma_20) / ma_20
        features['price_vs_ma50'] = (prices - ma_50) / ma_50
        
        return features.dropna()
    
    def fit(self, prices: pd.Series):
        """Fit the regime detection model"""
        features = self.extract_features(prices)
        features_scaled = self.scaler.fit_transform(features)
        self.gmm.fit(features_scaled)
        return self
    
    def predict_regime(self, prices: pd.Series) -> int:
        """Predict current market regime"""
        features = self.extract_features(prices)
        if len(features) == 0:
            return 2  # Default to sideways if insufficient data
        
        features_scaled = self.scaler.transform(features.iloc[-1:])
        regime = self.gmm.predict(features_scaled)[0]
        return regime

class BacktestEngine:
    """Comprehensive backtesting engine for RSI-based strategies"""
    
    def __init__(self, target_ticker: str = "TQQQ"):
        self.target_ticker = target_ticker
        self.results = {}
        self.trades = pd.DataFrame()
        self.equity_curve = pd.Series()
        self.performance_metrics = {}
        
    def calculate_rsi(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Calculate RSI using talib"""
        return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
    
    def run_backtest(self, prices: pd.Series, asset_prices: Dict[str, pd.Series], 
                    strategy_signals: pd.DataFrame, initial_capital: float = 100000) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            prices: Target ticker price series
            asset_prices: Dictionary of all asset price series
            strategy_signals: DataFrame with columns ['regime', 'rsi_condition', 'target_asset']
            initial_capital: Starting capital
            
        Returns:
            Dictionary with comprehensive backtest results
        """
        
        # Ensure all price series are aligned
        common_dates = prices.index
        for ticker, price_series in asset_prices.items():
            common_dates = common_dates.intersection(price_series.index)
        
        # Align data
        aligned_prices = prices.loc[common_dates]
        aligned_asset_prices = {ticker: series.loc[common_dates] 
                               for ticker, series in asset_prices.items()}
        aligned_signals = strategy_signals.loc[common_dates]
        
        # Initialize backtest variables
        capital = initial_capital
        positions = []
        equity_curve = []
        trades_list = []
        current_asset = None
        entry_price = None
        entry_date = None
        
        for date in aligned_signals.index[1:]:  # Start from second day
            signal = aligned_signals.loc[date]
            recommended_asset = signal['target_asset']
            
            # Check if we need to switch assets
            if current_asset != recommended_asset:
                # Close current position if exists
                if current_asset is not None and entry_price is not None:
                    exit_price = aligned_asset_prices[current_asset].loc[date]
                    if not pd.isna(exit_price) and exit_price > 0:
                        # Calculate return
                        asset_return = (exit_price - entry_price) / entry_price
                        new_capital = capital * (1 + asset_return)
                        
                        # Record trade
                        trades_list.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'asset': current_asset,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'return': asset_return,
                            'capital_before': capital,
                            'capital_after': new_capital,
                            'regime': signal['regime'],
                            'rsi_condition': signal['rsi_condition']
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
        
        # Close final position
        if current_asset is not None and entry_price is not None:
            final_date = aligned_signals.index[-1]
            exit_price = aligned_asset_prices[current_asset].loc[final_date]
            if not pd.isna(exit_price) and exit_price > 0:
                asset_return = (exit_price - entry_price) / entry_price
                final_capital = capital * (1 + asset_return)
                
                trades_list.append({
                    'entry_date': entry_date,
                    'exit_date': final_date,
                    'asset': current_asset,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': asset_return,
                    'capital_before': capital,
                    'capital_after': final_capital,
                    'regime': aligned_signals.loc[final_date]['regime'],
                    'rsi_condition': aligned_signals.loc[final_date]['rsi_condition']
                })
                
                capital = final_capital
            
            equity_curve.append(capital)
        
        # Store results
        self.trades = pd.DataFrame(trades_list)
        self.equity_curve = pd.Series(equity_curve, 
                                     index=aligned_signals.index[1:len(equity_curve)+1])
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(
            initial_capital, aligned_signals.index
        )
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics,
            'final_capital': capital,
            'total_return': (capital - initial_capital) / initial_capital
        }
    
    def _calculate_performance_metrics(self, initial_capital: float, date_index: pd.DatetimeIndex) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.equity_curve) == 0:
            return {}
        
        # Basic metrics
        total_return = (self.equity_curve.iloc[-1] - initial_capital) / initial_capital
        
        # Calculate daily returns
        daily_returns = self.equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {'total_return': total_return}
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if len(self.trades) > 0:
            win_rate = (self.trades['return'] > 0).mean()
            avg_win = self.trades[self.trades['return'] > 0]['return'].mean()
            avg_loss = self.trades[self.trades['return'] < 0]['return'].mean()
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else float('inf')
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(daily_returns)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        if not self.performance_metrics:
            print("No backtest results available. Run backtest first.")
            return
        
        print("="*60)
        print("BACKTEST PERFORMANCE REPORT")
        print("="*60)
        print(f"Target Ticker: {self.target_ticker}")
        print(f"Total Return: {self.performance_metrics['total_return']:.2%}")
        print(f"Annualized Return: {self.performance_metrics['annualized_return']:.2%}")
        print(f"Volatility: {self.performance_metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {self.performance_metrics['win_rate']:.2%}")
        print(f"Total Trades: {self.performance_metrics['total_trades']}")
        print(f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
        
        if len(self.trades) > 0:
            print(f"\nTrade Analysis:")
            print(f"Average Win: {self.performance_metrics['avg_win']:.2%}")
            print(f"Average Loss: {self.performance_metrics['avg_loss']:.2%}")
            
            # Regime-based performance
            regime_performance = self.trades.groupby('regime')['return'].agg(['count', 'mean', 'std'])
            print(f"\nPerformance by Market Regime:")
            for regime in regime_performance.index:
                stats = regime_performance.loc[regime]
                print(f"  {regime}: {stats['count']} trades, "
                      f"{stats['mean']:.2%} avg return, {stats['std']:.2%} volatility")

class RSISignalOptimizer:
    """Optimizes RSI signals and generates trading recommendations"""
    
    def __init__(self, regime_detector: MarketRegimeDetector, target_ticker: str = "TQQQ"):
        self.regime_detector = regime_detector
        self.target_ticker = target_ticker
        self.optimal_thresholds = {}
        self.asset_allocation_rules = {}
        
    def set_asset_allocation_rules(self, allocation_rules: Dict):
        """Set asset allocation rules for different RSI conditions and regimes"""
        self.asset_allocation_rules = allocation_rules
    
    def calculate_rsi(self, prices: pd.Series, window: int = 10) -> pd.Series:
        """Calculate RSI using talib"""
        return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
    
    def optimize_thresholds(self, prices: pd.Series, asset_prices: Dict[str, pd.Series]) -> Dict:
        """Optimize RSI thresholds for each market regime"""
        print(f"Optimizing RSI thresholds for {self.target_ticker} using {len(prices)} days of data...")
        
        # Calculate regime occurrences
        regime_counts = {}
        min_history = max(50, len(prices) // 20)
        
        for i in range(min_history, len(prices)):
            try:
                regime = self.regime_detector.predict_regime(prices.iloc[:i+1])
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            except:
                continue
        
        print(f"Regime distribution:")
        for regime, count in regime_counts.items():
            regime_name = self.regime_detector.regime_labels[regime]
            percentage = (count / sum(regime_counts.values())) * 100
            print(f"  {regime_name}: {count} days ({percentage:.1f}%)")
        
        # Optimize for each regime
        for regime in range(self.regime_detector.n_regimes):
            if regime not in regime_counts or regime_counts[regime] < 20:
                self.optimal_thresholds[regime] = {'lower': 30.0, 'upper': 70.0}
                continue
            
            lower, upper = self._optimize_regime_thresholds(prices, asset_prices, regime)
            self.optimal_thresholds[regime] = {'lower': lower, 'upper': upper}
            
            regime_name = self.regime_detector.regime_labels[regime]
            print(f"  {regime_name}: Lower={lower:.1f}, Upper={upper:.1f}")
        
        return self.optimal_thresholds
    
    def _optimize_regime_thresholds(self, prices: pd.Series, asset_prices: Dict[str, pd.Series], 
                                   regime: int) -> Tuple[float, float]:
        """Optimize thresholds for a specific regime using backtesting"""
        
        def objective(params):
            lower_thresh, upper_thresh = params
            
            # Generate signals for this threshold combination
            signals = self._generate_signals(prices, asset_prices, 
                                           {regime: {'lower': lower_thresh, 'upper': upper_thresh}})
            
            # Run quick backtest
            backtest = BacktestEngine(self.target_ticker)
            try:
                results = backtest.run_backtest(prices, asset_prices, signals)
                return -results['performance_metrics'].get('sharpe_ratio', -999)
            except:
                return 999  # Penalty for failed backtest
        
        # Optimization bounds and constraints
        bounds = [(10, 45), (55, 90)]
        constraints = {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 10}
        
        result = minimize(objective, [30, 70], method='L-BFGS-B', 
                         bounds=bounds, constraints=constraints)
        
        return (result.x[0], result.x[1]) if result.success else (30.0, 70.0)
    
    def _generate_signals(self, prices: pd.Series, asset_prices: Dict[str, pd.Series], 
                         thresholds: Dict) -> pd.DataFrame:
        """Generate trading signals based on RSI and regime"""
        rsi = self.calculate_rsi(prices)
        signals = []
        
        min_history = max(50, len(prices) // 20)
        
        for i, date in enumerate(prices.index):
            if i < min_history:
                signals.append({
                    'regime': 'Unknown',
                    'rsi_condition': 'Neutral',
                    'target_asset': self.target_ticker
                })
                continue
            
            try:
                # Detect current regime
                current_regime = self.regime_detector.predict_regime(prices.iloc[:i+1])
                regime_name = self.regime_detector.regime_labels[current_regime].lower()
                
                # Get thresholds for this regime
                if current_regime in thresholds:
                    lower = thresholds[current_regime]['lower']
                    upper = thresholds[current_regime]['upper']
                else:
                    lower, upper = 30, 70
                
                # Determine RSI condition
                current_rsi = rsi.iloc[i]
                if pd.isna(current_rsi):
                    rsi_condition = 'Neutral'
                elif current_rsi > upper:
                    rsi_condition = 'Overbought'
                elif current_rsi < lower:
                    rsi_condition = 'Oversold'
                else:
                    rsi_condition = 'Neutral'
                
                # Get target asset
                target_asset = self._get_target_asset(rsi_condition, regime_name)
                
                signals.append({
                    'regime': regime_name.title(),
                    'rsi_condition': rsi_condition,
                    'target_asset': target_asset,
                    'rsi_value': current_rsi,
                    'rsi_lower': lower,
                    'rsi_upper': upper
                })
                
            except:
                signals.append({
                    'regime': 'Unknown',
                    'rsi_condition': 'Neutral',
                    'target_asset': self.target_ticker
                })
        
        return pd.DataFrame(signals, index=prices.index)
    
    def _get_target_asset(self, rsi_condition: str, regime: str) -> str:
        """Get target asset based on RSI condition and regime"""
        condition_key = f'rsi_{rsi_condition.lower()}'
        
        if (condition_key in self.asset_allocation_rules and 
            regime in self.asset_allocation_rules[condition_key]):
            return self.asset_allocation_rules[condition_key][regime]
        
        return self.target_ticker
    
    def generate_strategy_signals(self, prices: pd.Series, asset_prices: Dict[str, pd.Series]) -> pd.DataFrame:
        """Generate complete strategy signals using optimized thresholds"""
        return self._generate_signals(prices, asset_prices, self.optimal_thresholds)

def load_or_simulate_data(simulate: bool = True, tickers: List[str] = None) -> Dict[str, pd.Series]:
    """Load real data or generate simulated data for backtesting"""
    
    if simulate:
        print("Generating simulated market data...")
        
        # Extended simulation with multiple market cycles
        np.random.seed(42)
        n_days = 2520  # ~10 years
        
        # Create correlated assets with different characteristics
        base_returns = np.random.normal(0.0005, 0.02, n_days)  # Base market
        
        assets = {
            'TQQQ': {'beta': 3.0, 'vol_mult': 1.8, 'drift': 0.0008},     # 3x Tech
            'TECL': {'beta': 3.0, 'vol_mult': 1.9, 'drift': 0.0009},     # 3x Tech Alt
            'SOXL': {'beta': 3.0, 'vol_mult': 2.1, 'drift': 0.0010},     # 3x Semiconductors
            'UVXY': {'beta': -1.5, 'vol_mult': 3.0, 'drift': -0.0015},   # VIX
            'SQQQ': {'beta': -3.0, 'vol_mult': 1.8, 'drift': -0.0008},   # Inverse 3x
            'BSV': {'beta': 0.1, 'vol_mult': 0.3, 'drift': 0.0002}       # Short-term bonds
        }
        
        price_data = {}
        start_date = pd.Timestamp.now() - pd.Timedelta(days=n_days*1.4)
        dates = pd.date_range(start_date, periods=n_days, freq='B')
        
        for ticker, params in assets.items():
            # Generate correlated returns
            asset_returns = (base_returns * params['beta'] + 
                           np.random.normal(params['drift'], 0.01, n_days)) * params['vol_mult']
            
            # Convert to prices
            prices = [100]  # Starting price
            for ret in asset_returns:
                prices.append(prices[-1] * (1 + ret))
            
            price_data[ticker] = pd.Series(prices[1:], index=dates)
        
        return price_data
    
    else:
        # In real implementation, would load data from API/database
        print("Real data loading not implemented in this example")
        return {}

# Main execution - Standalone backtest engine
if __name__ == "__main__":
    print("="*70)
    print("RSI BACKTEST ENGINE & SIGNAL OPTIMIZER")
    print("="*70)
    
    # Configuration
    TARGET_TICKER = "TQQQ"
    INITIAL_CAPITAL = 100000
    
    # Asset allocation strategy
    ALLOCATION_RULES = {
        'rsi_overbought': {
            'bull': 'UVXY',      # VIX when overbought in bull
            'bear': 'SQQQ',      # Inverse when overbought in bear  
            'sideways': 'BSV'    # Bonds when overbought in sideways
        },
        'rsi_oversold': {
            'bull': 'TECL',      # Tech 3x when oversold in bull
            'bear': 'TQQQ',      # Regular when oversold in bear
            'sideways': 'SOXL'   # Semiconductor 3x when oversold in sideways
        },
        'rsi_neutral': {
            'bull': 'TQQQ',      # Default position
            'bear': 'BSV',       # Conservative in bear
            'sideways': 'TQQQ'   # Default position
        }
    }
    
    # Load/generate data
    print(f"Setting up backtest for target ticker: {TARGET_TICKER}")
    asset_data = load_or_simulate_data(simulate=True)
    target_data = asset_data[TARGET_TICKER]
    
    print(f"Data range: {target_data.index[0].strftime('%Y-%m-%d')} to {target_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total trading days: {len(target_data)}")
    
    # Initialize components
    regime_detector = MarketRegimeDetector(n_regimes=3)
    regime_detector.fit(target_data)
    
    signal_optimizer = RSISignalOptimizer(regime_detector, TARGET_TICKER)
    signal_optimizer.set_asset_allocation_rules(ALLOCATION_RULES)
    
    # Optimize thresholds
    print(f"\nOptimizing RSI thresholds...")
    optimal_thresholds = signal_optimizer.optimize_thresholds(target_data, asset_data)
    
    # Generate strategy signals
    print(f"Generating strategy signals...")
    strategy_signals = signal_optimizer.generate_strategy_signals(target_data, asset_data)
    
    # Run backtest
    print(f"Running backtest with ${INITIAL_CAPITAL:,} initial capital...")
    backtest_engine = BacktestEngine(TARGET_TICKER)
    backtest_results = backtest_engine.run_backtest(
        target_data, asset_data, strategy_signals, INITIAL_CAPITAL
    )
    
    # Print results
    backtest_engine.print_performance_report()
    
    print(f"\n" + "="*70)
    print("STRATEGY SUMMARY")
    print("="*70)
    print(f"Target Ticker: {TARGET_TICKER}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Final Capital: ${backtest_results['final_capital']:,.2f}")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    
    print(f"\nOptimized RSI Thresholds:")
    for regime_id, thresholds in optimal_thresholds.items():
        regime_name = regime_detector.regime_labels[regime_id]
        print(f"  {regime_name}: {thresholds['lower']:.1f} / {thresholds['upper']:.1f}")
    
    print(f"\nAsset Allocation Rules:")
    for condition, rules in ALLOCATION_RULES.items():
        print(f"  {condition.replace('_', ' ').title()}:")
        for regime, asset in rules.items():
            print(f"    {regime.title()} -> {asset}")
    
    print(f"\nStrategy Performance:")
    print(f"  Sharpe Ratio: {backtest_results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['performance_metrics']['max_drawdown']:.2%}")
    print(f"  Win Rate: {backtest_results['performance_metrics']['win_rate']:.2%}")
    print(f"  Total Trades: {backtest_results['performance_metrics']['total_trades']}")
    
    # Sample recent signals
    print(f"\n" + "="*70)
    print("RECENT SIGNALS (Last 5 Days)")
    print("="*70)
    recent_signals = strategy_signals.tail()
    for date, signal in recent_signals.iterrows():
        print(f"{date.strftime('%Y-%m-%d')}: "
              f"{signal['regime']} market, RSI {signal['rsi_condition']} "
              f"({signal['rsi_value']:.1f}) -> Hold {signal['target_asset']}")
    
    print(f"\n" + "="*70)
    print("CURRENT RECOMMENDATION")
    print("="*70)
    latest_signal = strategy_signals.iloc[-1]
    print(f"Market Regime: {latest_signal['regime']}")
    print(f"RSI Value: {latest_signal['rsi_value']:.1f}")
    print(f"RSI Condition: {latest_signal['rsi_condition']}")
    print(f"Recommended Asset: {latest_signal['target_asset']}")
    print(f"Dynamic Thresholds: {latest_signal['rsi_lower']:.1f} / {latest_signal['rsi_upper']:.1f}")
