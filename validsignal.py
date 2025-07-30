import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import talib
from typing import Dict, Tuple, List
import warnings
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

class RSIThresholdOptimizer:
    """Optimizes RSI thresholds based on market regimes with target ticker functionality"""
    
    def __init__(self, regime_detector: MarketRegimeDetector, target_ticker: str = "TQQQ"):
        self.regime_detector = regime_detector
        self.optimal_thresholds = {}
        self.min_data_points = 50  # Minimum data points needed for regime detection
        self.use_all_available_data = True  # Use maximum lookback available
        self.target_ticker = target_ticker  # Primary ticker to analyze RSI for
        self.asset_allocation_rules = {}  # Store optimal asset allocation per regime/condition
        
    def set_asset_allocation_rules(self, allocation_rules: Dict):
        """
        Set asset allocation rules for different RSI conditions and regimes
        
        Args:
            allocation_rules: Dictionary defining which assets to hold under different conditions
            Example:
            {
                'rsi_overbought': {
                    'bull': 'UVXY',      # Hold VIX when overbought in bull market
                    'bear': 'SQQQ',      # Hold inverse when overbought in bear market  
                    'sideways': 'BSV'    # Hold bonds when overbought in sideways market
                },
                'rsi_oversold': {
                    'bull': 'TECL',      # Hold tech bull 3x when oversold in bull market
                    'bear': 'TQQQ',      # Hold regular when oversold in bear market
                    'sideways': 'TQQQ'   # Hold regular when oversold in sideways market
                },
                'rsi_neutral': {
                    'bull': 'TQQQ',      # Default position
                    'bear': 'BSV',       # Conservative in bear market
                    'sideways': 'TQQQ'   # Default position
                }
            }
        """
        self.asset_allocation_rules = allocation_rules
    
    def get_target_asset(self, rsi_value: float, market_regime: str, 
                        rsi_lower: float, rsi_upper: float) -> str:
        """
        Determine which asset to hold based on RSI value and market regime
        
        Args:
            rsi_value: Current RSI value for target ticker
            market_regime: Current market regime ('bull', 'bear', 'sideways')
            rsi_lower: Lower RSI threshold
            rsi_upper: Upper RSI threshold
            
        Returns:
            Asset ticker symbol to hold
        """
        if not self.asset_allocation_rules:
            return self.target_ticker  # Default to target ticker if no rules set
        
        # Determine RSI condition
        if rsi_value > rsi_upper:
            condition = 'rsi_overbought'
        elif rsi_value < rsi_lower:
            condition = 'rsi_oversold'
        else:
            condition = 'rsi_neutral'
        
        # Get asset for this condition and regime
        if condition in self.asset_allocation_rules:
            regime_rules = self.asset_allocation_rules[condition]
            if market_regime.lower() in regime_rules:
                return regime_rules[market_regime.lower()]
        
        # Fallback to target ticker
        return self.target_ticker
        """Calculate RSI using talib"""
        return talib.RSI(prices.values, timeperiod=window)
    
    def backtest_strategy(self, prices: pd.Series, rsi_upper: float, rsi_lower: float, 
                         regime: int) -> float:
        """Backtest RSI strategy for a specific regime with given thresholds using all available data"""
        rsi = self.calculate_rsi(prices)
        
        # Use all available data - determine regime for each day with sufficient history
        regime_mask = []
        min_history = max(50, len(prices) // 20)  # At least 50 days or 5% of total data
        
        for i in range(len(prices)):
            if i < min_history:  # Need enough data for regime detection
                regime_mask.append(False)
                continue
            try:
                # Use expanding window - all data from start to current point
                current_regime = self.regime_detector.predict_regime(prices.iloc[:i+1])
                regime_mask.append(current_regime == regime)
            except:
                regime_mask.append(False)
        
        regime_mask = pd.Series(regime_mask, index=prices.index)
        
        # Generate signals only for the specific regime across entire dataset
        signals = pd.Series(0, index=prices.index)
        
        for i in range(1, len(rsi)):
            if not regime_mask.iloc[i] or pd.isna(rsi[i]):
                continue
                
            # RSI-based signals
            if rsi[i] > rsi_upper:
                signals.iloc[i] = -1  # Sell/Short signal
            elif rsi[i] < rsi_lower:
                signals.iloc[i] = 1   # Buy signal
            else:
                signals.iloc[i] = 0   # Hold
        
        # Calculate returns using all available data
        returns = prices.pct_change()
        strategy_returns = signals.shift(1) * returns  # Signal from previous day
        
        # Only consider returns when in the specific regime
        regime_returns = strategy_returns[regime_mask]
        
        if len(regime_returns) == 0 or regime_returns.sum() == 0:
            return -999  # Penalty for no valid trades
        
        # Calculate enhanced performance metrics using all available data
        total_days = len(regime_returns[regime_returns != 0])
        if total_days < 10:  # Need minimum trades for reliable statistics
            return -999
        
        # Calculate annualized Sharpe ratio
        mean_return = regime_returns.mean() * 252
        std_return = regime_returns.std() * np.sqrt(252)
        
        if std_return == 0:
            return -999
        
        sharpe = mean_return / std_return
        
        # Bonus for using more historical data (encourages longer backtests)
        data_bonus = min(0.1, total_days / 1000)  # Small bonus for more data points
        sharpe += data_bonus
        
        # Add penalty for extreme thresholds to encourage reasonable values
        if rsi_upper > 95 or rsi_lower < 5 or rsi_upper - rsi_lower < 10:
            sharpe -= 1
        
        return sharpe
    
    def optimize_thresholds_for_regime(self, prices: pd.Series, regime: int) -> Tuple[float, float]:
        """Optimize RSI thresholds for a specific market regime"""
        
        def objective(params):
            rsi_lower, rsi_upper = params
            return -self.backtest_strategy(prices, rsi_upper, rsi_lower, regime)
        
        # Initial guess and bounds
        initial_guess = [30, 70]  # Standard RSI thresholds
        bounds = [(10, 45), (55, 90)]  # Reasonable bounds for RSI thresholds
        
        # Add constraint that upper > lower + 10
        constraints = {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 10}
        
        result = minimize(objective, initial_guess, method='L-BFGS-B', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x[0], result.x[1]
        else:
            # Return default values if optimization fails
            return 30, 70
    
    def fit(self, prices: pd.Series):
        """Optimize thresholds for all market regimes using maximum available historical data"""
        total_days = len(prices)
        print(f"Optimizing RSI thresholds using {total_days} days of historical data...")
        print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
        
        # Calculate regime statistics across entire dataset
        regime_counts = {}
        min_history = max(50, len(prices) // 20)
        
        # Count regime occurrences across all available data
        for i in range(min_history, len(prices)):
            try:
                regime = self.regime_detector.predict_regime(prices.iloc[:i+1])
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            except:
                continue
        
        print(f"Regime distribution across {total_days - min_history} trading days:")
        for regime, count in regime_counts.items():
            regime_name = self.regime_detector.regime_labels[regime]
            percentage = (count / sum(regime_counts.values())) * 100
            print(f"  {regime_name}: {count} days ({percentage:.1f}%)")
        
        # Optimize thresholds for each regime using all available data
        for regime in range(self.regime_detector.n_regimes):
            regime_name = self.regime_detector.regime_labels[regime]
            print(f"\nOptimizing for regime {regime} ({regime_name}) using full dataset...")
            
            if regime not in regime_counts or regime_counts[regime] < 20:
                print(f"  Insufficient data for regime {regime}, using default thresholds")
                self.optimal_thresholds[regime] = {'lower': 30.0, 'upper': 70.0}
                continue
            
            lower, upper = self.optimize_thresholds_for_regime(prices, regime)
            self.optimal_thresholds[regime] = {
                'lower': round(lower, 1),
                'upper': round(upper, 1)
            }
            
            print(f"  Regime {regime} ({regime_name}): Lower={lower:.1f}, Upper={upper:.1f}")
            print(f"  Based on {regime_counts[regime]} trading days in this regime")
        
        return self
    
    def get_current_thresholds(self, prices: pd.Series) -> Dict[str, float]:
        """Get optimal thresholds for current market regime"""
        current_regime = self.regime_detector.predict_regime(prices)
        
        if current_regime in self.optimal_thresholds:
            thresholds = self.optimal_thresholds[current_regime]
            regime_name = self.regime_detector.regime_labels[current_regime]
            
            return {
                'regime': regime_name,
                'regime_id': current_regime,
                'rsi_lower': thresholds['lower'],
                'rsi_upper': thresholds['upper']
            }
        else:
            # Fallback to default thresholds
            return {
                'regime': 'Unknown',
                'regime_id': -1,
                'rsi_lower': 30.0,
                'rsi_upper': 70.0
            }

# Example usage with extended historical data simulation
def simulate_extended_market_data(n_years=10):
    """Generate extended simulated market data for comprehensive testing"""
    np.random.seed(42)
    
    # Create longer market cycles with more varied regimes
    total_days = n_years * 252  # Trading days per year
    
    # Define multiple market cycles with different characteristics
    cycles = [
        {'type': 'bull', 'days': 500, 'drift': 0.0008, 'vol': 0.018},      # Strong bull
        {'type': 'bear', 'days': 300, 'drift': -0.0012, 'vol': 0.035},    # Bear crash
        {'type': 'recovery', 'days': 400, 'drift': 0.0006, 'vol': 0.025}, # Volatile recovery
        {'type': 'sideways', 'days': 350, 'drift': 0.0001, 'vol': 0.015}, # Range-bound
        {'type': 'bull', 'days': 600, 'drift': 0.0007, 'vol': 0.020},     # Sustained bull
        {'type': 'correction', 'days': 200, 'drift': -0.0008, 'vol': 0.028}, # Correction
        {'type': 'sideways', 'days': 300, 'drift': -0.0001, 'vol': 0.012}, # Low vol sideways
        {'type': 'bull', 'days': 370, 'drift': 0.0009, 'vol': 0.022},     # Final bull run
    ]
    
    prices = []
    current_price = 100
    cycle_info = []
    
    for cycle in cycles:
        if len(prices) >= total_days:
            break
            
        days_remaining = min(cycle['days'], total_days - len(prices))
        daily_returns = np.random.normal(cycle['drift'], cycle['vol'], days_remaining)
        
        cycle_start = len(prices)
        for ret in daily_returns:
            current_price *= (1 + ret)
            prices.append(current_price)
        
        cycle_info.append({
            'type': cycle['type'],
            'start': cycle_start,
            'end': len(prices) - 1,
            'days': days_remaining
        })
    
    # Create date range starting from earlier date for more history
    start_date = pd.Timestamp.now() - pd.Timedelta(days=n_years*365)
    dates = pd.date_range(start_date, periods=len(prices), freq='B')  # Business days
    
    return pd.Series(prices, index=dates), cycle_info

# Main execution with target ticker functionality
if __name__ == "__main__":
    # Generate extended market data (10 years worth)
    print("Generating 10 years of simulated market data for maximum lookback...")
    market_data, cycle_info = simulate_extended_market_data(n_years=10)
    
    print(f"Generated {len(market_data)} trading days of data")
    print(f"Date range: {market_data.index[0].strftime('%Y-%m-%d')} to {market_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Price range: ${market_data.min():.2f} to ${market_data.max():.2f}")
    
    print("\nMarket cycles in simulated data:")
    for i, cycle in enumerate(cycle_info):
        start_date = market_data.index[cycle['start']].strftime('%Y-%m-%d')
        end_date = market_data.index[cycle['end']].strftime('%Y-%m-%d')
        print(f"  {i+1}. {cycle['type'].title()}: {start_date} to {end_date} ({cycle['days']} days)")
    
    # Initialize and train the regime detector using all available data
    print(f"\nTraining market regime detector on full {len(market_data)} day dataset...")
    regime_detector = MarketRegimeDetector(n_regimes=3)
    regime_detector.fit(market_data)
    
    # Initialize optimizer with target ticker
    target_ticker = "TQQQ"  # User can change this
    print(f"\nInitializing RSI threshold optimizer for target ticker: {target_ticker}")
    threshold_optimizer = RSIThresholdOptimizer(regime_detector, target_ticker=target_ticker)
    
    # Define asset allocation rules (user can customize this)
    allocation_rules = {
        'rsi_overbought': {
            'bull': 'UVXY',      # VIX when overbought in bull market
            'bear': 'SQQQ',      # Inverse QQQ when overbought in bear market  
            'sideways': 'BSV'    # Bonds when overbought in sideways market
        },
        'rsi_oversold': {
            'bull': 'TECL',      # Tech bull 3x when oversold in bull market
            'bear': 'TQQQ',      # Regular QQQ when oversold in bear market
            'sideways': 'SOXL'   # Semiconductor bull when oversold in sideways
        },
        'rsi_neutral': {
            'bull': 'TQQQ',      # Default position in bull
            'bear': 'BSV',       # Conservative bonds in bear market
            'sideways': 'TQQQ'   # Default position in sideways
        }
    }
    
    print("\nSetting up asset allocation rules:")
    for condition, rules in allocation_rules.items():
        print(f"  {condition.replace('_', ' ').title()}:")
        for regime, asset in rules.items():
            print(f"    {regime.title()} market -> {asset}")
    
    threshold_optimizer.set_asset_allocation_rules(allocation_rules)
    
    # Optimize thresholds using maximum lookback
    threshold_optimizer.fit(market_data)
    
    # Get complete current recommendation
    print("\n" + "="*70)
    print("CURRENT TRADING RECOMMENDATION (Target Ticker Analysis)")
    print("="*70)
    
    recommendation = threshold_optimizer.get_current_recommendation(market_data)
    
    print(f"Target Ticker: {recommendation['target_ticker']}")
    print(f"Current RSI: {recommendation['current_rsi']}")
    print(f"RSI Condition: {recommendation['rsi_condition']}")
    print(f"Market Regime: {recommendation['market_regime']}")
    print(f"Dynamic Thresholds: {recommendation['rsi_lower_threshold']}/{recommendation['rsi_upper_threshold']}")
    print(f"Recommended Asset: {recommendation['recommended_asset']}")
    print(f"Reasoning: {recommendation['reasoning']}")
    
    # Show all regime thresholds with asset allocations
    print("\n" + "="*70)
    print("COMPLETE ASSET ALLOCATION STRATEGY BY REGIME")
    print("="*70)
    
    for regime_id, thresholds in threshold_optimizer.optimal_thresholds.items():
        regime_name = regime_detector.regime_labels[regime_id].lower()
        print(f"\n{regime_name.title()} Market (Regime {regime_id}):")
        print(f"  RSI Thresholds: {thresholds['lower']} / {thresholds['upper']}")
        print(f"  Asset Allocation:")
        
        # Show what happens in each RSI condition
        sample_rsi_values = [25, 50, 85]  # Oversold, neutral, overbought
        for rsi_val in sample_rsi_values:
            asset = threshold_optimizer.get_target_asset(
                rsi_val, regime_name, thresholds['lower'], thresholds['upper']
            )
            condition = "Oversold" if rsi_val < thresholds['lower'] else \
                       "Overbought" if rsi_val > thresholds['upper'] else "Neutral"
            print(f"    RSI {rsi_val} ({condition}) -> {asset}")
    
    # Generate Composer symphony code with target ticker functionality
    print("\n" + "="*70)
    print("COMPOSER SYMPHONY WITH TARGET TICKER FUNCTIONALITY")
    print("="*70)
    
    print(f"""
;; Dynamic RSI strategy with target ticker analysis
;; Target: {target_ticker}
;; Current regime: {recommendation['market_regime']}
;; Current thresholds: {recommendation['rsi_lower_threshold']}/{recommendation['rsi_upper_threshold']}

(defsymphony
 "Target Ticker Dynamic RSI Strategy"
 {{:asset-class "EQUITIES", :rebalance-threshold 0.05}}
 (weight-equal
  [(cond
    ;; Overbought condition - current threshold: {recommendation['rsi_upper_threshold']}
    (> (rsi "{target_ticker}" {{:window 10}}) (get-dynamic-rsi-upper))
    [(asset (get-overbought-asset) "Asset for overbought {target_ticker}")]
    
    ;; Oversold condition - current threshold: {recommendation['rsi_lower_threshold']}  
    (< (rsi "{target_ticker}" {{:window 10}}) (get-dynamic-rsi-lower))
    [(asset (get-oversold-asset) "Asset for oversold {target_ticker}")]
    
    ;; Neutral condition
    :else
    [(asset (get-neutral-asset) "Default asset for neutral {target_ticker}")])]))

;; Current recommendations based on market regime:
;; Market regime: {recommendation['market_regime']}
;; If {target_ticker} RSI > {recommendation['rsi_upper_threshold']}: Hold {allocation_rules['rsi_overbought'][recommendation['market_regime'].lower()]}
;; If {target_ticker} RSI < {recommendation['rsi_lower_threshold']}: Hold {allocation_rules['rsi_oversold'][recommendation['market_regime'].lower()]}  
;; If neutral: Hold {allocation_rules['rsi_neutral'][recommendation['market_regime'].lower()]}

;; Example with current conditions:
;; Current RSI: {recommendation['current_rsi']} ({recommendation['rsi_condition']})
;; -> Recommended asset: {recommendation['recommended_asset']}
    """)
    
    print("\n" + "="*70) 
    print("USER CUSTOMIZATION OPTIONS")
    print("="*70)
    print(f"""
To customize for your strategy:

1. Change target ticker:
   threshold_optimizer = RSIThresholdOptimizer(regime_detector, target_ticker="SPY")

2. Modify asset allocation rules:
   allocation_rules = {{
       'rsi_overbought': {{'bull': 'VXX', 'bear': 'SPXS', 'sideways': 'TLT'}},
       'rsi_oversold': {{'bull': 'UPRO', 'bear': 'SPY', 'sideways': 'QQQ'}},
       'rsi_neutral': {{'bull': 'SPY', 'bear': 'TLT', 'sideways': 'SPY'}}
   }}

3. Current setup analyzes: {target_ticker}
4. Uses {len(market_data)} days of historical data
5. Adapts thresholds based on market regime
6. Recommends specific assets for each condition

Current recommendation: Hold {recommendation['recommended_asset']} 
({recommendation['reasoning']})
    """)
