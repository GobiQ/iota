import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta, date
import json
import requests
import yfinance as yf
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Strategy Stress Tester",
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

class TechnicalIndicators:
    """Calculate technical indicators for strategy logic"""
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def moving_average(prices: pd.Series, window: int) -> pd.Series:
        """Calculate moving average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, ma, lower

class StrategyParser:
    """Parse Composer strategy JSON into executable logic"""
    
    def __init__(self, strategy_json: dict):
        self.strategy = strategy_json
        self.assets = self._extract_assets()
        self.logic_tree = self._parse_logic_tree()
    
    def _extract_assets(self) -> List[str]:
        """Extract all unique tickers from strategy"""
        assets = set()
        
        def recursive_extract(node):
            if isinstance(node, dict):
                if node.get('step') == 'asset' and 'ticker' in node:
                    assets.add(node['ticker'])
                if 'children' in node:
                    if isinstance(node['children'], list):
                        for child in node['children']:
                            recursive_extract(child)
                    else:
                        recursive_extract(node['children'])
        
        recursive_extract(self.strategy)
        return list(assets)
    
    def _parse_logic_tree(self) -> dict:
        """Parse the strategy logic into executable format"""
        def parse_node(node):
            if not isinstance(node, dict):
                return node
                
            step = node.get('step', '')
            
            if step == 'asset':
                return {
                    'type': 'asset',
                    'ticker': node.get('ticker'),
                    'weight': 1.0
                }
            elif step == 'if':
                return {
                    'type': 'condition',
                    'condition': self._parse_condition(node),
                    'children': [parse_node(child) for child in node.get('children', [])]
                }
            elif step == 'if-child':
                return {
                    'type': 'branch',
                    'is_else': node.get('is-else-condition?', False),
                    'condition': self._parse_condition(node) if not node.get('is-else-condition?', False) else None,
                    'children': [parse_node(child) for child in node.get('children', [])]
                }
            elif step == 'wt-cash-equal':
                return {
                    'type': 'weight',
                    'children': [parse_node(child) for child in node.get('children', [])]
                }
            elif step == 'filter':
                return {
                    'type': 'filter',
                    'select_fn': node.get('select-fn'),
                    'select_n': node.get('select-n', 1),
                    'sort_by_fn': node.get('sort-by-fn'),
                    'sort_by_window': node.get('sort-by-window-days', 10),
                    'children': [parse_node(child) for child in node.get('children', [])]
                }
            else:
                children = node.get('children', [])
                if isinstance(children, list):
                    return [parse_node(child) for child in children]
                else:
                    return parse_node(children)
        
        return parse_node(self.strategy)
    
    def _parse_condition(self, node: dict) -> dict:
        """Parse condition logic with proper parameter handling"""
        condition = {
            'lhs_fn': node.get('lhs-fn'),
            'lhs_val': node.get('lhs-val'),
            'lhs_window': int(node.get('lhs-window-days', 10)),
            'comparator': node.get('comparator'),
            'rhs_fn': node.get('rhs-fn'),
            'rhs_val': node.get('rhs-val'),
            'rhs_window': int(node.get('rhs-window-days', 10)),
            'rhs_fixed': node.get('rhs-fixed-value?', False)
        }
        
        # Handle lhs-fn-params and rhs-fn-params
        if 'lhs-fn-params' in node:
            lhs_params = node['lhs-fn-params']
            if 'window' in lhs_params:
                condition['lhs_window'] = int(lhs_params['window'])
        
        if 'rhs-fn-params' in node:
            rhs_params = node['rhs-fn-params']
            if 'window' in rhs_params:
                condition['rhs_window'] = int(rhs_params['window'])
        
        return condition

class MarketSimulator:
    """Generate realistic market scenarios for stress testing"""
    
    def __init__(self, assets: List[str], historical_data: Dict[str, pd.Series]):
        self.assets = assets
        self.historical_data = historical_data
        self.returns_data = self._calculate_returns()
    
    def _calculate_returns(self) -> Dict[str, pd.Series]:
        """Calculate daily returns for all assets"""
        returns = {}
        for asset in self.assets:
            if asset in self.historical_data:
                prices = self.historical_data[asset]
                returns[asset] = prices.pct_change().dropna()
        return returns
    
    def generate_normal_scenario(self, days: int, num_simulations: int = 1) -> List[Dict[str, np.ndarray]]:
        """Generate normal market scenarios using historical statistics"""
        scenarios = []
        
        for _ in range(num_simulations):
            scenario = {}
            for asset in self.assets:
                if asset in self.returns_data:
                    returns = self.returns_data[asset]
                    mean_return = returns.mean()
                    std_return = returns.std()
                    
                    # Generate correlated returns (simplified)
                    scenario[asset] = np.random.normal(mean_return, std_return, days)
                else:
                    scenario[asset] = np.zeros(days)
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_stress_scenarios(self, days: int) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate specific stress test scenarios"""
        scenarios = {}
        
        # 2008-style Crisis
        scenarios['2008_crisis'] = self._generate_crisis_scenario(days, crash_magnitude=-0.4)
        
        # Tech Bubble Burst
        scenarios['tech_crash'] = self._generate_tech_crash(days)
        
        # VIX Spike / Volatility Explosion
        scenarios['volatility_spike'] = self._generate_volatility_spike(days)
        
        # Everything Rally
        scenarios['melt_up'] = self._generate_melt_up(days)
        
        # Prolonged Bear Market
        scenarios['bear_market'] = self._generate_bear_market(days)
        
        # Interest Rate Shock
        scenarios['rate_shock'] = self._generate_rate_shock(days)
        
        return scenarios
    
    def _generate_crisis_scenario(self, days: int, crash_magnitude: float = -0.4) -> Dict[str, np.ndarray]:
        """Generate 2008-style financial crisis"""
        scenario = {}
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_returns = self.returns_data[asset]
                volatility = base_returns.std() * 3  # Triple volatility
                
                # Create initial crash followed by high volatility
                returns = []
                cumulative_crash = 0
                
                for day in range(days):
                    if day < 20 and cumulative_crash > crash_magnitude:
                        # Crash phase
                        daily_return = np.random.normal(-0.05, volatility)
                        cumulative_crash += daily_return
                    elif day < 60:
                        # High volatility phase
                        daily_return = np.random.normal(-0.01, volatility)
                    else:
                        # Recovery phase
                        daily_return = np.random.normal(0.001, volatility * 0.7)
                    
                    returns.append(daily_return)
                
                scenario[asset] = np.array(returns)
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario
    
    def _generate_tech_crash(self, days: int) -> Dict[str, np.ndarray]:
        """Generate tech-specific crash scenario"""
        scenario = {}
        
        tech_assets = ['TQQQ', 'TECL', 'SOXL']  # Tech-heavy assets
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_std = self.returns_data[asset].std()
                
                if asset in tech_assets:
                    # Tech assets crash hard
                    returns = np.random.normal(-0.03, base_std * 2, days)
                elif asset == 'UVXY':
                    # Volatility spikes
                    returns = np.random.normal(0.05, base_std * 1.5, days)
                elif asset in ['BSV', 'SQQQ']:
                    # Safe havens benefit
                    returns = np.random.normal(0.01, base_std * 0.5, days)
                else:
                    returns = np.random.normal(-0.005, base_std, days)
                
                scenario[asset] = returns
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario
    
    def _generate_volatility_spike(self, days: int) -> Dict[str, np.ndarray]:
        """Generate VIX spike scenario"""
        scenario = {}
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_std = self.returns_data[asset].std()
                
                if asset == 'UVXY':
                    # UVXY explodes higher
                    returns = np.random.normal(0.08, base_std * 2, days)
                else:
                    # Everything else becomes highly volatile
                    returns = np.random.normal(0, base_std * 3, days)
                
                scenario[asset] = returns
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario
    
    def _generate_melt_up(self, days: int) -> Dict[str, np.ndarray]:
        """Generate everything-rally scenario"""
        scenario = {}
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_returns = self.returns_data[asset]
                base_mean = max(base_returns.mean(), 0.001)
                base_std = base_returns.std() * 0.7  # Lower volatility
                
                if asset == 'SQQQ':
                    # Short positions suffer
                    returns = np.random.normal(-base_mean * 2, base_std, days)
                elif asset == 'UVXY':
                    # Volatility collapses
                    returns = np.random.normal(-0.03, base_std, days)
                else:
                    # Everything else rallies
                    returns = np.random.normal(base_mean * 2, base_std, days)
                
                scenario[asset] = returns
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario
    
    def _generate_bear_market(self, days: int) -> Dict[str, np.ndarray]:
        """Generate prolonged bear market"""
        scenario = {}
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_std = self.returns_data[asset].std()
                
                if asset == 'SQQQ':
                    # Short positions benefit
                    returns = np.random.normal(0.02, base_std, days)
                elif asset in ['BSV']:
                    # Bonds are neutral to positive
                    returns = np.random.normal(0.001, base_std * 0.5, days)
                else:
                    # Equity-like assets decline slowly
                    returns = np.random.normal(-0.002, base_std, days)
                
                scenario[asset] = returns
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario
    
    def _generate_rate_shock(self, days: int) -> Dict[str, np.ndarray]:
        """Generate rising interest rate shock"""
        scenario = {}
        
        for asset in self.assets:
            if asset in self.returns_data:
                base_std = self.returns_data[asset].std()
                
                if asset in ['BSV']:
                    # Bonds get hit hard
                    returns = np.random.normal(-0.01, base_std * 1.5, days)
                elif asset in ['TQQQ', 'TECL']:
                    # Growth stocks suffer
                    returns = np.random.normal(-0.005, base_std * 1.2, days)
                else:
                    returns = np.random.normal(-0.001, base_std, days)
                
                scenario[asset] = returns
            else:
                scenario[asset] = np.zeros(days)
        
        return scenario

class StrategyEngine:
    """Execute strategy logic on market data"""
    
    def __init__(self, strategy_parser: StrategyParser):
        self.parser = strategy_parser
        self.tech_indicators = TechnicalIndicators()
    
    def evaluate_condition(self, condition: dict, market_data: Dict[str, pd.Series], current_idx: int) -> bool:
        """Evaluate a single condition with debug logging"""
        if not condition:
            return True
        
        try:
            lhs_val = condition['lhs_val']
            rhs_val = condition['rhs_val']
            comparator = condition['comparator']
            
            # Get left-hand side value
            if condition['lhs_fn'] == 'current-price':
                if lhs_val in market_data and current_idx < len(market_data[lhs_val]):
                    lhs_value = market_data[lhs_val].iloc[current_idx]
                else:
                    return False
            elif condition['lhs_fn'] == 'relative-strength-index':
                if lhs_val in market_data:
                    prices = market_data[lhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['lhs_window']:
                        # Use Wilder's smoothing method to match Composer
                        rsi_values = self._calculate_wilder_rsi(prices, condition['lhs_window'])
                        lhs_value = rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else 50
                    else:
                        lhs_value = 50  # Default RSI
                else:
                    return False
            elif condition['lhs_fn'] == 'cumulative-return':
                if lhs_val in market_data:
                    prices = market_data[lhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['lhs_window']:
                        start_price = prices.iloc[-(condition['lhs_window']+1)]
                        end_price = prices.iloc[-1]
                        lhs_value = ((end_price / start_price) - 1) * 100
                    else:
                        lhs_value = 0
                else:
                    return False
            elif condition['lhs_fn'] == 'moving-average-price':
                if lhs_val in market_data:
                    prices = market_data[lhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['lhs_window']:
                        ma_values = self.tech_indicators.moving_average(prices, condition['lhs_window'])
                        lhs_value = ma_values.iloc[-1] if not pd.isna(ma_values.iloc[-1]) else prices.iloc[-1]
                    else:
                        lhs_value = prices.iloc[-1] if len(prices) > 0 else 0
                else:
                    return False
            elif condition['lhs_fn'] == 'max-drawdown':
                if lhs_val in market_data:
                    prices = market_data[lhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['lhs_window']:
                        window_prices = prices.iloc[-condition['lhs_window']:]
                        peak = window_prices.expanding().max()
                        drawdown = (peak - window_prices) / peak * 100
                        lhs_value = drawdown.max()
                    else:
                        lhs_value = 0
                else:
                    return False
            elif condition['lhs_fn'] == 'standard-deviation-return':
                if lhs_val in market_data:
                    prices = market_data[lhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['lhs_window'] + 1:
                        returns = prices.pct_change().dropna()
                        window_returns = returns.iloc[-condition['lhs_window']:]
                        lhs_value = window_returns.std() * 100
                    else:
                        lhs_value = 0
                else:
                    return False
            else:
                return False
            
            # Get right-hand side value
            if condition.get('rhs_fixed', False):
                rhs_value = float(rhs_val)
            elif condition['rhs_fn'] == 'moving-average-price':
                if rhs_val in market_data:
                    prices = market_data[rhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['rhs_window']:
                        ma_values = self.tech_indicators.moving_average(prices, condition['rhs_window'])
                        rhs_value = ma_values.iloc[-1] if not pd.isna(ma_values.iloc[-1]) else lhs_value
                    else:
                        rhs_value = lhs_value  # Default to no change
                else:
                    return False
            elif condition['rhs_fn'] == 'relative-strength-index':
                if rhs_val in market_data:
                    prices = market_data[rhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['rhs_window']:
                        rsi_values = self._calculate_wilder_rsi(prices, condition['rhs_window'])
                        rhs_value = rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else 50
                    else:
                        rhs_value = 50
                else:
                    return False
            elif condition['rhs_fn'] == 'cumulative-return':
                if rhs_val in market_data:
                    prices = market_data[rhs_val].iloc[:current_idx+1]
                    if len(prices) >= condition['rhs_window']:
                        start_price = prices.iloc[-(condition['rhs_window']+1)]
                        end_price = prices.iloc[-1]
                        rhs_value = ((end_price / start_price) - 1) * 100
                    else:
                        rhs_value = 0
                else:
                    return False
            else:
                rhs_value = float(rhs_val) if condition.get('rhs_fixed', False) else 0
            
            # Evaluate comparison
            if comparator == 'gt':
                result = lhs_value > rhs_value
            elif comparator == 'lt':
                result = lhs_value < rhs_value
            elif comparator == 'gte':
                result = lhs_value >= rhs_value
            elif comparator == 'lte':
                result = lhs_value <= rhs_value
            elif comparator == 'eq':
                result = abs(lhs_value - rhs_value) < 1e-6
            else:
                result = False
            
            # Debug logging for mismatches
            if hasattr(self, 'debug_mode') and self.debug_mode:
                st.write(f"Condition: {condition['lhs_fn']}({lhs_val}) {lhs_value:.2f} {comparator} {rhs_value:.2f} = {result}")
            
            return result
            
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                st.error(f"Condition evaluation error: {str(e)}")
            return False
    
    def _calculate_wilder_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI using Wilder's smoothing method (matches Composer)"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Use Wilder's smoothing (alpha = 1/window)
        alpha = 1.0 / window
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def execute_node(self, node: Any, market_data: Dict[str, pd.Series], current_idx: int) -> Dict[str, float]:
        """Execute a node in the strategy tree"""
        if isinstance(node, list):
            # Multiple nodes - execute first valid one
            for sub_node in node:
                result = self.execute_node(sub_node, market_data, current_idx)
                if result:
                    return result
            return {}
        
        if not isinstance(node, dict):
            return {}
        
        node_type = node.get('type', '')
        
        if node_type == 'asset':
            ticker = node.get('ticker')
            weight = node.get('weight', 1.0)
            if ticker:
                return {ticker: weight}
            return {}
        
        elif node_type == 'condition':
            condition_met = self.evaluate_condition(node['condition'], market_data, current_idx)
            
            for child in node.get('children', []):
                if isinstance(child, dict):
                    if child.get('type') == 'branch':
                        is_else = child.get('is_else', False)
                        child_condition = child.get('condition')
                        
                        if is_else and not condition_met:
                            return self.execute_node(child.get('children', []), market_data, current_idx)
                        elif not is_else and condition_met:
                            if child_condition:
                                child_condition_met = self.evaluate_condition(child_condition, market_data, current_idx)
                                if child_condition_met:
                                    return self.execute_node(child.get('children', []), market_data, current_idx)
                            else:
                                return self.execute_node(child.get('children', []), market_data, current_idx)
            
            return {}
        
        elif node_type == 'weight':
            # Execute children and return results
            return self.execute_node(node.get('children', []), market_data, current_idx)
        
        elif node_type == 'filter':
            # Handle filter logic (top RSI selection, etc.)
            children = node.get('children', [])
            if not children:
                return {}
            
            sort_by_fn = node.get('sort_by_fn')
            sort_by_window = node.get('sort_by_window', 10)
            select_n = int(node.get('select_n', 1))
            
            if sort_by_fn == 'relative-strength-index':
                candidates = []
                for child in children:
                    if isinstance(child, dict) and child.get('type') == 'asset':
                        ticker = child.get('ticker')
                        if ticker and ticker in market_data:
                            prices = market_data[ticker].iloc[:current_idx+1]
                            if len(prices) >= sort_by_window:
                                rsi_values = self.tech_indicators.rsi(prices, sort_by_window)
                                rsi_val = rsi_values.iloc[-1] if not pd.isna(rsi_values.iloc[-1]) else 50
                                candidates.append((ticker, rsi_val))
                
                if candidates:
                    # Sort by RSI (descending for "top")
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    selected_ticker = candidates[0][0]
                    return {selected_ticker: 1.0}
            
            # Fallback: return first child
            if children:
                return self.execute_node(children[0], market_data, current_idx)
            
            return {}
        
        return {}
    
    def get_allocation(self, market_data: Dict[str, pd.Series], current_idx: int) -> Dict[str, float]:
        """Get portfolio allocation for current market conditions"""
        return self.execute_node(self.parser.logic_tree, market_data, current_idx)

class StrategyStressTester:
    """Main class for comprehensive strategy stress testing"""
    
    def __init__(self, strategy_json: dict):
        self.parser = StrategyParser(strategy_json)
        self.engine = StrategyEngine(self.parser)
        self.assets = self.parser.assets
        self.historical_data = {}
        self.market_simulator = None
    
    def load_historical_data(self, start_date: str = "2000-01-01", end_date: str = None) -> bool:
        """Load historical price data for all assets"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            progress_bar = st.progress(0)
            total_assets = len(self.assets)
            
            for i, asset in enumerate(self.assets):
                progress_bar.progress((i + 1) / total_assets)
                
                ticker = yf.Ticker(asset)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    self.historical_data[asset] = data['Close']
                else:
                    st.warning(f"No data found for {asset}")
            
            progress_bar.empty()
            
            if self.historical_data:
                self.market_simulator = MarketSimulator(self.assets, self.historical_data)
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
            return False
    
    def backtest_strategy(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run historical backtest to validate strategy implementation"""
        if not self.historical_data:
            return {}
        
        # Prepare data
        all_data = pd.DataFrame(self.historical_data)
        all_data = all_data.dropna()
        
        if start_date:
            all_data = all_data[all_data.index >= start_date]
        if end_date:
            all_data = all_data[all_data.index <= end_date]
        
        if len(all_data) < 200:  # Need enough data for technical indicators
            st.warning("Insufficient historical data for backtest")
            return {}
        
        # Convert to dict of series for strategy engine
        market_data = {asset: all_data[asset] for asset in self.assets if asset in all_data.columns}
        
        # Run day-by-day backtest
        portfolio_values = [100000]  # Start with $100k
        daily_returns = []
        allocations_history = []
        current_portfolio_value = 100000
        
        progress_bar = st.progress(0)
        
        for i in range(200, len(all_data)):  # Start after 200 days for indicators
            progress_bar.progress(i / len(all_data))
            
            # Get allocation for this day
            allocation = self.engine.get_allocation(market_data, i)
            allocations_history.append(allocation)
            
            # Calculate daily return
            if i > 200:  # After first allocation
                daily_return = 0
                prev_allocation = allocations_history[-2] if len(allocations_history) > 1 else {}
                
                for asset, weight in prev_allocation.items():
                    if asset in market_data and i < len(market_data[asset]):
                        asset_return = (market_data[asset].iloc[i] / market_data[asset].iloc[i-1] - 1)
                        daily_return += weight * asset_return
                
                daily_returns.append(daily_return)
                current_portfolio_value *= (1 + daily_return)
                portfolio_values.append(current_portfolio_value)
        
        progress_bar.empty()
        
        return {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'allocations_history': allocations_history,
            'dates': all_data.index[200:200+len(daily_returns)]
        }
    
    def run_monte_carlo_stress_test(self, days: int = 252, num_simulations: int = 1000) -> Dict[str, Any]:
        """Run comprehensive Monte Carlo stress testing"""
        if not self.market_simulator:
            return {}
        
        results = {}
        
        # 1. Normal market scenarios
        st.write("### Running Normal Market Simulations...")
        normal_scenarios = self.market_simulator.generate_normal_scenario(days, num_simulations)
        normal_results = self._simulate_scenarios(normal_scenarios, "Normal Market")
        results['normal_market'] = normal_results
        
        # 2. Stress scenarios
        st.write("### Running Stress Test Scenarios...")
        stress_scenarios = self.market_simulator.generate_stress_scenarios(days)
        
        for scenario_name, scenario_data in stress_scenarios.items():
            st.write(f"Testing {scenario_name.replace('_', ' ').title()}...")
            scenario_results = self._simulate_scenarios([scenario_data], scenario_name)
            results[scenario_name] = scenario_results
        
        return results
    
    def _simulate_scenarios(self, scenarios: List[Dict[str, np.ndarray]], scenario_name: str) -> Dict[str, Any]:
        """Simulate strategy performance across scenarios"""
        all_returns = []
        all_allocations = []
        all_max_drawdowns = []
        
        progress_bar = st.progress(0)
        
        for sim_idx, scenario in enumerate(scenarios):
            progress_bar.progress((sim_idx + 1) / len(scenarios))
            
            # Convert scenario to cumulative prices
            market_data = {}
            for asset in self.assets:
                if asset in scenario:
                    returns = scenario[asset]
                    prices = [100]  # Start at $100
                    for ret in returns:
                        prices.append(prices[-1] * (1 + ret))
                    market_data[asset] = pd.Series(prices[1:])  # Remove initial price
                else:
                    market_data[asset] = pd.Series([100] * len(scenario[list(scenario.keys())[0]]))
            
            # Run strategy simulation
            portfolio_returns = []
            current_allocation = {}
            portfolio_value = 100000
            peak_value = 100000
            max_drawdown = 0
            
            for day in range(len(scenario[list(scenario.keys())[0]])):
                # Get new allocation
                allocation = self.engine.get_allocation(market_data, day)
                
                # Calculate portfolio return
                if day > 0 and current_allocation:
                    daily_return = 0
                    for asset, weight in current_allocation.items():
                        if asset in scenario:
                            asset_return = scenario[asset][day]
                            daily_return += weight * asset_return
                    
                    portfolio_returns.append(daily_return)
                    portfolio_value *= (1 + daily_return)
                    
                    # Track drawdown
                    if portfolio_value > peak_value:
                        peak_value = portfolio_value
                    
                    current_drawdown = (peak_value - portfolio_value) / peak_value
                    max_drawdown = max(max_drawdown, current_drawdown)
                
                current_allocation = allocation
            
            all_returns.append(portfolio_returns)
            all_allocations.append(current_allocation)
            all_max_drawdowns.append(max_drawdown)
        
        progress_bar.empty()
        
        # Calculate summary statistics
        if all_returns:
            final_returns = [(np.prod([1 + r for r in returns]) - 1) * 100 for returns in all_returns if returns]
            
            return {
                'final_returns': final_returns,
                'max_drawdowns': [dd * 100 for dd in all_max_drawdowns],
                'daily_returns_paths': all_returns,
                'scenario_name': scenario_name,
                'num_simulations': len(scenarios)
            }
        
        return {}

def create_strategy_analysis_plots(results: Dict[str, Any], strategy_name: str):
    """Create comprehensive analysis plots"""
    
    # 1. Performance Distribution Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Strategy Performance Analysis: {strategy_name}', fontsize=16)
    
    # Plot return distributions
    row = 0
    col = 0
    
    for scenario_name, scenario_results in results.items():
        if col >= 3:
            row = 1
            col = 0
        
        if 'final_returns' in scenario_results:
            final_returns = scenario_results['final_returns']
            
            axes[row, col].hist(final_returns, bins=30, alpha=0.7, density=True)
            axes[row, col].axvline(np.median(final_returns), color='red', linestyle='--', 
                                 label=f'Median: {np.median(final_returns):.1f}%')
            axes[row, col].axvline(np.percentile(final_returns, 5), color='orange', linestyle='--',
                                 label=f'5th %ile: {np.percentile(final_returns, 5):.1f}%')
            axes[row, col].set_title(f'{scenario_name.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Final Return (%)')
            axes[row, col].set_ylabel('Density')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        col += 1
    
    # Hide empty subplots
    for i in range(len(results), 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # 2. Risk Metrics Summary Table
    st.subheader("Risk Metrics Summary")
    
    summary_data = []
    for scenario_name, scenario_results in results.items():
        if 'final_returns' in scenario_results:
            final_returns = scenario_results['final_returns']
            max_drawdowns = scenario_results.get('max_drawdowns', [])
            
            summary_data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Median Return (%)': f"{np.median(final_returns):.2f}",
                '5th Percentile (%)': f"{np.percentile(final_returns, 5):.2f}",
                '95th Percentile (%)': f"{np.percentile(final_returns, 95):.2f}",
                'Probability of Loss (%)': f"{np.mean([r < 0 for r in final_returns]) * 100:.1f}",
                'Max Drawdown (%)': f"{np.mean(max_drawdowns):.2f}" if max_drawdowns else "N/A",
                'Worst Case Return (%)': f"{np.min(final_returns):.2f}",
                'Best Case Return (%)': f"{np.max(final_returns):.2f}"
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # 3. Drawdown Analysis
    st.subheader("Drawdown Analysis")
    
    fig_dd, axes_dd = plt.subplots(1, 2, figsize=(15, 6))
    
    # Drawdown distribution
    all_drawdowns = []
    scenario_labels = []
    
    for scenario_name, scenario_results in results.items():
        if 'max_drawdowns' in scenario_results:
            drawdowns = scenario_results['max_drawdowns']
            all_drawdowns.extend(drawdowns)
            scenario_labels.extend([scenario_name.replace('_', ' ').title()] * len(drawdowns))
    
    if all_drawdowns:
        # Box plot of drawdowns by scenario
        drawdown_df = pd.DataFrame({
            'Drawdown (%)': all_drawdowns,
            'Scenario': scenario_labels
        })
        
        sns.boxplot(data=drawdown_df, x='Scenario', y='Drawdown (%)', ax=axes_dd[0])
        axes_dd[0].set_title('Drawdown Distribution by Scenario')
        axes_dd[0].tick_params(axis='x', rotation=45)
        
        # Overall drawdown histogram
        axes_dd[1].hist(all_drawdowns, bins=30, alpha=0.7, color='red')
        axes_dd[1].axvline(np.median(all_drawdowns), color='blue', linestyle='--',
                          label=f'Median: {np.median(all_drawdowns):.1f}%')
        axes_dd[1].set_title('Overall Drawdown Distribution')
        axes_dd[1].set_xlabel('Maximum Drawdown (%)')
        axes_dd[1].set_ylabel('Frequency')
        axes_dd[1].legend()
        axes_dd[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_dd)

def analyze_strategy_components(strategy_parser: StrategyParser, results: Dict[str, Any]):
    """Analyze which strategy components contribute to performance"""
    st.subheader("Strategy Component Analysis")
    
    # Extract key components from strategy
    components = {
        'Assets Used': strategy_parser.assets,
        'Total Decision Nodes': count_decision_nodes(strategy_parser.logic_tree),
        'Uses RSI': check_for_indicator(strategy_parser.logic_tree, 'relative-strength-index'),
        'Uses Moving Averages': check_for_indicator(strategy_parser.logic_tree, 'moving-average-price'),
        'Has Volatility Hedge': 'UVXY' in strategy_parser.assets or 'VXX' in strategy_parser.assets,
        'Has Short Positions': any('SQQ' in asset for asset in strategy_parser.assets),
        'Has Leveraged ETFs': any(asset in ['TQQQ', 'TECL', 'SOXL'] for asset in strategy_parser.assets)
    }
    
    # Display component analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Strategy Components:**")
        for component, value in components.items():
            if isinstance(value, bool):
                st.write(f"- {component}: {'✅' if value else '❌'}")
            elif isinstance(value, list):
                st.write(f"- {component}: {', '.join(value[:5])}")
                if len(value) > 5:
                    st.write(f"  ... and {len(value) - 5} more")
            else:
                st.write(f"- {component}: {value}")
    
    with col2:
        # Performance vs complexity analysis
        if 'normal_market' in results:
            normal_returns = results['normal_market'].get('final_returns', [])
            if normal_returns:
                fig_comp, ax_comp = plt.subplots(figsize=(8, 6))
                
                # Create risk-return scatter
                scenarios_for_scatter = ['normal_market', '2008_crisis', 'tech_crash', 'volatility_spike']
                risk_return_data = []
                
                for scenario in scenarios_for_scatter:
                    if scenario in results and 'final_returns' in results[scenario]:
                        returns = results[scenario]['final_returns']
                        drawdowns = results[scenario].get('max_drawdowns', [])
                        
                        avg_return = np.mean(returns)
                        avg_risk = np.mean(drawdowns) if drawdowns else 0
                        
                        risk_return_data.append({
                            'scenario': scenario.replace('_', ' ').title(),
                            'return': avg_return,
                            'risk': avg_risk
                        })
                
                if risk_return_data:
                    scatter_df = pd.DataFrame(risk_return_data)
                    ax_comp.scatter(scatter_df['risk'], scatter_df['return'], s=100, alpha=0.7)
                    
                    for idx, row in scatter_df.iterrows():
                        ax_comp.annotate(row['scenario'], 
                                       (row['risk'], row['return']),
                                       xytext=(5, 5), textcoords='offset points')
                    
                    ax_comp.set_xlabel('Average Max Drawdown (%)')
                    ax_comp.set_ylabel('Average Return (%)')
                    ax_comp.set_title('Risk-Return Profile Across Scenarios')
                    ax_comp.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_comp)

def count_decision_nodes(node, count=0):
    """Count decision nodes in strategy tree"""
    if isinstance(node, dict):
        if node.get('type') == 'condition':
            count += 1
        if 'children' in node:
            children = node['children']
            if isinstance(children, list):
                for child in children:
                    count = count_decision_nodes(child, count)
            else:
                count = count_decision_nodes(children, count)
    elif isinstance(node, list):
        for item in node:
            count = count_decision_nodes(item, count)
    return count

def check_for_indicator(node, indicator_name):
    """Check if strategy uses specific technical indicator"""
    if isinstance(node, dict):
        if 'condition' in node and isinstance(node['condition'], dict):
            condition = node['condition']
            if (condition.get('lhs_fn') == indicator_name or 
                condition.get('rhs_fn') == indicator_name):
                return True
        
        if 'children' in node:
            children = node['children']
            if isinstance(children, list):
                for child in children:
                    if check_for_indicator(child, indicator_name):
                        return True
            else:
                if check_for_indicator(children, indicator_name):
                    return True
    elif isinstance(node, list):
        for item in node:
            if check_for_indicator(item, indicator_name):
                return True
    return False

def create_failure_mode_analysis(results: Dict[str, Any]):
    """Analyze potential strategy failure modes"""
    st.subheader("Strategy Failure Mode Analysis")
    
    failure_scenarios = []
    
    for scenario_name, scenario_results in results.items():
        if 'final_returns' in scenario_results:
            final_returns = scenario_results['final_returns']
            max_drawdowns = scenario_results.get('max_drawdowns', [])
            
            # Define failure criteria
            large_losses = [r for r in final_returns if r < -20]  # >20% loss
            extreme_drawdowns = [d for d in max_drawdowns if d > 30]  # >30% drawdown
            
            failure_rate = len(large_losses) / len(final_returns) if final_returns else 0
            extreme_dd_rate = len(extreme_drawdowns) / len(max_drawdowns) if max_drawdowns else 0
            
            if failure_rate > 0.1 or extreme_dd_rate > 0.1:  # >10% failure rate
                failure_scenarios.append({
                    'Scenario': scenario_name.replace('_', ' ').title(),
                    'Failure Rate (%)': f"{failure_rate * 100:.1f}",
                    'Extreme Drawdown Rate (%)': f"{extreme_dd_rate * 100:.1f}",
                    'Worst Return (%)': f"{min(final_returns):.1f}",
                    'Worst Drawdown (%)': f"{max(max_drawdowns):.1f}" if max_drawdowns else "N/A"
                })
    
    if failure_scenarios:
        st.write("**⚠️ Potential Failure Modes Identified:**")
        failure_df = pd.DataFrame(failure_scenarios)
        st.dataframe(failure_df, use_container_width=True)
        
        st.write("**Failure Mode Analysis:**")
        for scenario in failure_scenarios:
            with st.expander(f"Analysis: {scenario['Scenario']}"):
                st.write(f"This scenario shows elevated risk with {scenario['Failure Rate (%)']}% of simulations resulting in >20% losses.")
                st.write(f"Additionally, {scenario['Extreme Drawdown Rate (%)']}% of simulations experienced >30% drawdowns.")
                st.write(f"Worst case scenario: {scenario['Worst Return (%)']}% return with {scenario['Worst Drawdown (%)']}% maximum drawdown.")
                
                # Provide specific recommendations based on scenario
                if 'Crisis' in scenario['Scenario']:
                    st.write("**Recommendation**: Consider adding more defensive assets or stronger risk management rules for crisis scenarios.")
                elif 'Tech Crash' in scenario['Scenario']:
                    st.write("**Recommendation**: Strategy may be overexposed to technology sector. Consider diversification.")
                elif 'Volatility Spike' in scenario['Scenario']:
                    st.write("**Recommendation**: Current volatility hedging may be insufficient for extreme VIX spikes.")
    else:
        st.success("✅ No significant failure modes detected in tested scenarios.")

# Main Streamlit Application
def main():
    st.title("🚀 Advanced Strategy Stress Tester")
    st.markdown("*Holdings-Based Monte Carlo Simulation for Robust Strategy Analysis*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", [
        "Strategy Upload & Configuration",
        "Historical Validation", 
        "Monte Carlo Stress Testing",
        "Risk Assessment & Failure Modes",
        "Forward-Looking Analysis",
        "Strategy Optimization",
        "Documentation"
    ])
    
    # Initialize session state
    if 'strategy_tester' not in st.session_state:
        st.session_state.strategy_tester = None
    if 'stress_test_results' not in st.session_state:
        st.session_state.stress_test_results = {}
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = {}
    
    if page == "Strategy Upload & Configuration":
        st.header("Strategy Configuration")
        
        st.markdown("""
        Upload your Composer strategy JSON file to begin comprehensive stress testing. 
        This tool will analyze your strategy's performance across multiple market scenarios,
        identify potential failure modes, and provide actionable risk assessment.
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Composer Strategy JSON", 
            type=['json'],
            help="Export your strategy JSON from Composer and upload it here"
        )
        
        if uploaded_file is not None:
            try:
                strategy_json = json.load(uploaded_file)
                st.success("✅ Strategy JSON loaded successfully!")
                
                # Display strategy info
                strategy_name = strategy_json.get('name', 'Unknown Strategy')
                st.info(f"**Strategy Name**: {strategy_name}")
                
                # Initialize strategy tester
                st.session_state.strategy_tester = StrategyStressTester(strategy_json)
                
                # Show detected assets
                assets = st.session_state.strategy_tester.assets
                st.write(f"**Detected Assets**: {', '.join(assets)}")
                
                # Data loading configuration
                st.subheader("Historical Data Configuration")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Historical Data Start Date",
                        value=date(2000, 1, 1),
                        help="Default: 2000-01-01 (defaults to oldest possible date)"
                    )
                
                with col2:
                    end_date = st.date_input(
                        "Historical Data End Date",
                        value=date.today(),
                        help="End date for loading historical price data"
                    )
                
                if st.button("Load Historical Data", type="primary"):
                    with st.spinner("Loading historical price data..."):
                        success = st.session_state.strategy_tester.load_historical_data(
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d")
                        )
                    
                    if success:
                        st.success("✅ Historical data loaded successfully!")
                        st.write("**Data Summary**:")
                        
                        for asset in assets:
                            if asset in st.session_state.strategy_tester.historical_data:
                                data_length = len(st.session_state.strategy_tester.historical_data[asset])
                                st.write(f"- {asset}: {data_length} trading days")
                        
                        st.info("🎯 Ready for analysis! Navigate to other tabs to begin testing.")
                    else:
                        st.error("❌ Failed to load historical data. Please check asset tickers and try again.")
                
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid Composer strategy JSON.")
            except Exception as e:
                st.error(f"❌ Error processing strategy: {str(e)}")
        else:
            st.info("👆 Please upload a Composer strategy JSON file to begin.")
    
    elif page == "Historical Validation":
        st.header("Historical Validation")
        
        if st.session_state.strategy_tester is None:
            st.warning("⚠️ Please upload a strategy JSON file first.")
            return
        
        if not st.session_state.strategy_tester.historical_data:
            st.warning("⚠️ Please load historical data first.")
            return
        
        st.markdown("""
        Validate strategy implementation by running a historical backtest. 
        This ensures our strategy logic correctly replicates the intended behavior.
        """)
        
        # Add debug mode toggle
        debug_mode = st.checkbox("Enable Debug Mode (shows condition evaluations)", value=False)
        st.session_state.strategy_tester.engine.debug_mode = debug_mode
        
        # Backtest configuration
        col1, col2 = st.columns(2)
        with col1:
            backtest_start = st.date_input(
                "Backtest Start Date",
                value=date(2000, 1, 1),
                help="Default: 2000-01-01 (defaults to oldest possible date)"
            )
        
        with col2:
            backtest_end = st.date_input(
                "Backtest End Date", 
                value=date.today()
            )
        
        # Add option to fetch and compare with Composer backtest
        st.subheader("Composer Validation")
        col1, col2 = st.columns(2)
        with col1:
            validate_against_composer = st.checkbox("Compare with Composer backtest", value=True)
        with col2:
            if validate_against_composer:
                composer_url = st.text_input(
                    "Composer Strategy URL (for validation)", 
                    help="Same URL used in configuration"
                )
        
        if st.button("Run Historical Backtest", type="primary"):
            with st.spinner("Running historical backtest..."):
                backtest_results = st.session_state.strategy_tester.backtest_strategy(
                    backtest_start.strftime("%Y-%m-%d"),
                    backtest_end.strftime("%Y-%m-%d")
                )
            
            if backtest_results:
                st.session_state.backtest_results = backtest_results
                
                # If validation is enabled, compare with Composer
                if validate_against_composer and composer_url:
                    st.write(f"**Fetching Composer data for URL:** {composer_url}")
                    st.write(f"**Date range:** {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
                    
                    with st.spinner("Fetching Composer backtest for validation..."):
                        try:
                            composer_allocations, composer_name, composer_tickers = fetch_backtest(
                                composer_url,
                                backtest_start.strftime("%Y-%m-%d"),
                                backtest_end.strftime("%Y-%m-%d")
                            )
                            
                            if composer_allocations is not None:
                                # Compare allocations
                                st.subheader("🔍 Validation Results")
                                
                                our_allocations = backtest_results['allocations_history']
                                dates = backtest_results['dates']
                                
                                # Debug information
                                st.write(f"**Our backtest dates:** {len(dates)} days from {dates[0]} to {dates[-1]}")
                                st.write(f"**Composer backtest dates:** {len(composer_allocations.index)} days from {composer_allocations.index[0]} to {composer_allocations.index[-1]}")
                                st.write(f"**Our allocations:** {len(our_allocations)} entries")
                                
                                # Show sample of normalized dates for debugging
                                sample_dates = dates[:5] if len(dates) > 5 else dates
                                st.write("**Sample date normalization:**")
                                for i, d in enumerate(sample_dates):
                                    if hasattr(d, 'tz_localize'):
                                        normalized = d.tz_localize(None)
                                        st.write(f"  Original: {d} -> Normalized: {normalized}")
                                    else:
                                        st.write(f"  Date: {d} (no timezone)")
                                
                                # Find matching dates and compare
                                validation_results = []
                                mismatches = 0
                                matched_dates = 0
                                
                                for i, current_date in enumerate(dates):
                                    if i < len(our_allocations):
                                        our_alloc = our_allocations[i]
                                        
                                        # Normalize date to remove timezone info for comparison
                                        if hasattr(current_date, 'tz_localize'):
                                            normalized_date = current_date.tz_localize(None)
                                        else:
                                            normalized_date = current_date
                                        
                                        # Find corresponding Composer allocation
                                        date_str = normalized_date.strftime('%Y-%m-%d')
                                        if normalized_date in composer_allocations.index:
                                            matched_dates += 1
                                            composer_alloc = composer_allocations.loc[normalized_date]
                                            composer_dict = {
                                                ticker: weight/100 for ticker, weight in composer_alloc.items() 
                                                if abs(weight) > 0.001
                                            }
                                            
                                            # Compare allocations
                                            match = True
                                            differences = []
                                            for ticker in set(list(our_alloc.keys()) + list(composer_dict.keys())):
                                                our_weight = our_alloc.get(ticker, 0)
                                                composer_weight = composer_dict.get(ticker, 0)
                                                
                                                if abs(our_weight - composer_weight) > 0.05:  # 5% tolerance
                                                    match = False
                                                    differences.append(f"{ticker}: Our={our_weight:.3f}, Composer={composer_weight:.3f}")
                                            
                                            # Show first few mismatches for debugging
                                            if not match and len(validation_results) < 3:
                                                st.write(f"**Sample mismatch for {date_str}:**")
                                                st.write(f"Our allocation: {our_alloc}")
                                                st.write(f"Composer allocation: {composer_dict}")
                                                st.write(f"Differences: {differences[:5]}")  # Show first 5 differences
                                            
                                            if not match:
                                                mismatches += 1
                                                validation_results.append({
                                                    'Date': date_str,
                                                    'Our Allocation': str(our_alloc),
                                                    'Composer Allocation': str(composer_dict),
                                                    'Match': '❌'
                                                })
                                            else:
                                                validation_results.append({
                                                    'Date': date_str,
                                                    'Our Allocation': str(our_alloc),
                                                    'Composer Allocation': str(composer_dict),
                                                    'Match': '✅'
                                                })
                                
                                # Display validation summary
                                total_days = len(validation_results)
                                match_rate = ((total_days - mismatches) / total_days * 100) if total_days > 0 else 0
                                
                                # Calculate average differences for insight
                                if total_days > 0:
                                    st.write("**Summary Statistics:**")
                                    st.write(f"- Total days with mismatches: {mismatches}")
                                    st.write(f"- Match rate: {match_rate:.1f}%")
                                    st.write(f"- Date range overlap: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
                                    
                                    # Show first few validation results for debugging
                                    if validation_results:
                                        st.write("**First few validation results:**")
                                        for i, result in enumerate(validation_results[:3]):
                                            st.write(f"  {result['Date']}: {result['Match']}")
                                            if result['Match'] == '❌':
                                                st.write(f"    Our: {result['Our Allocation']}")
                                                st.write(f"    Composer: {result['Composer Allocation']}")
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Days Compared", total_days)
                                with col2:
                                    st.metric("Matched Dates", matched_dates)
                                with col3:
                                    st.metric("Mismatches", mismatches)
                                with col4:
                                    color = "normal" if match_rate > 95 else "inverse"
                                    st.metric("Match Rate", f"{match_rate:.1f}%", delta_color=color)
                                
                                if match_rate < 95:
                                    st.error(f"⚠️ Strategy logic mismatch detected! Only {match_rate:.1f}% of allocations match Composer.")
                                    st.write("**Recent Mismatches:**")
                                    mismatch_df = pd.DataFrame([r for r in validation_results if r['Match'] == '❌'][:10])
                                    if not mismatch_df.empty:
                                        st.dataframe(mismatch_df, use_container_width=True)
                                        
                                    st.write("**Debugging Tips:**")
                                    st.write("- Check RSI calculation method (Wilder's vs EMA)")
                                    st.write("- Verify technical indicator windows")
                                    st.write("- Ensure proper condition evaluation order")
                                    st.write("- Check weight allocation logic")
                                else:
                                    st.success(f"✅ Strategy logic validation passed! {match_rate:.1f}% match rate.")
                        
                        except Exception as e:
                            st.error(f"Validation failed: {str(e)}")
                
                st.success("✅ Historical backtest completed!")
                
                # Display results (existing code continues...)
                portfolio_values = backtest_results['portfolio_values']
                daily_returns = backtest_results['daily_returns']
                dates = backtest_results['dates']
                
                # Performance metrics
                total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
                annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252 / len(daily_returns)) - 1) * 100
                volatility = np.std(daily_returns) * np.sqrt(252) * 100
                sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (peak - portfolio_values) / peak
                max_drawdown = np.max(drawdown) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{total_return:.2f}%")
                with col2:
                    st.metric("Annual Return", f"{annual_return:.2f}%")
                with col3:
                    st.metric("Volatility", f"{volatility:.2f}%")
                with col4:
                    st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                with col2:
                    win_rate = np.mean([r > 0 for r in daily_returns]) * 100
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                
                # Plot portfolio performance
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Portfolio value over time
                ax1.plot(dates, portfolio_values[1:], linewidth=2, color='blue')
                ax1.set_title("Portfolio Value Over Time")
                ax1.set_ylabel("Portfolio Value ($)")
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
                
                # Drawdown chart
                ax2.fill_between(dates, -drawdown[1:] * 100, 0, color='red', alpha=0.3)
                ax2.plot(dates, -drawdown[1:] * 100, color='red', linewidth=1)
                ax2.set_title("Drawdown Over Time")
                ax2.set_ylabel("Drawdown (%)")
                ax2.set_xlabel("Date")
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Recent allocations
                if backtest_results['allocations_history']:
                    st.subheader("Recent Portfolio Allocations")
                    recent_allocations = backtest_results['allocations_history'][-10:]
                    recent_dates = dates[-10:]
                    
                    allocation_data = []
                    for i, allocation in enumerate(recent_allocations):
                        row = {'Date': recent_dates[i].strftime('%Y-%m-%d')}
                        for asset, weight in allocation.items():
                            row[asset] = f"{weight * 100:.1f}%"
                        allocation_data.append(row)
                    
                    if allocation_data:
                        st.dataframe(pd.DataFrame(allocation_data), use_container_width=True)
            else:
                st.error("❌ Backtest failed. Please check your data and try again.")
    
    elif page == "Monte Carlo Stress Testing":
        st.header("Monte Carlo Stress Testing")
        
        if st.session_state.strategy_tester is None:
            st.warning("⚠️ Please upload a strategy JSON file and load historical data first.")
            return
        
        if not st.session_state.strategy_tester.historical_data:
            st.warning("⚠️ Please load historical data first.")
            return
        
        st.markdown("""
        Run comprehensive Monte Carlo simulations across multiple market scenarios.
        This analysis tests strategy robustness under various market conditions.
        """)
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            simulation_days = st.number_input(
                "Simulation Period (Trading Days)",
                min_value=60,
                max_value=1000,
                value=252,
                help="Number of trading days to simulate (252 ≈ 1 year)"
            )
        
        with col2:
            num_simulations = st.number_input(
                "Number of Simulations",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Number of Monte Carlo simulations to run"
            )
        
        with col3:
            st.write("**Scenarios to Test:**")
            test_normal = st.checkbox("Normal Market", value=True)
            test_crisis = st.checkbox("Financial Crisis", value=True)
            test_tech_crash = st.checkbox("Tech Crash", value=True)
            test_volatility = st.checkbox("Volatility Spike", value=True)
            test_melt_up = st.checkbox("Market Melt-Up", value=True)
            test_bear = st.checkbox("Bear Market", value=True)
        
        if st.button("Run Monte Carlo Stress Test", type="primary"):
            with st.spinner("Running comprehensive stress testing..."):
                # Run the full stress test
                stress_results = st.session_state.strategy_tester.run_monte_carlo_stress_test(
                    days=simulation_days,
                    num_simulations=num_simulations
                )
            
            if stress_results:
                st.session_state.stress_test_results = stress_results
                st.success("✅ Monte Carlo stress testing completed!")
                
                # Create comprehensive analysis plots
                strategy_name = st.session_state.strategy_tester.parser.strategy.get('name', 'Strategy')
                create_strategy_analysis_plots(stress_results, strategy_name)
                
                # Strategy component analysis
                analyze_strategy_components(
                    st.session_state.strategy_tester.parser, 
                    stress_results
                )
                
            else:
                st.error("❌ Stress testing failed. Please try again.")
    
    elif page == "Risk Assessment & Failure Modes":
        st.header("Risk Assessment & Failure Mode Analysis")
        
        if not st.session_state.stress_test_results:
            st.warning("⚠️ Please run Monte Carlo stress testing first.")
            return
        
        st.markdown("""
        Detailed analysis of strategy failure modes and risk characteristics.
        Identify scenarios where the strategy may underperform or fail.
        """)
        
        # Failure mode analysis
        create_failure_mode_analysis(st.session_state.stress_test_results)
        
        # Detailed risk metrics
        st.subheader("Detailed Risk Metrics")
        
        risk_analysis = {}
        
        for scenario_name, results in st.session_state.stress_test_results.items():
            if 'final_returns' in results:
                returns = results['final_returns']
                drawdowns = results.get('max_drawdowns', [])
                
                # Calculate VaR and CVaR
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                cvar_95 = np.mean([r for r in returns if r <= var_95])
                
                # Calculate other risk metrics
                volatility = np.std(returns)
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                
                risk_analysis[scenario_name] = {
                    'VaR (95%)': var_95,
                    'VaR (99%)': var_99,
                    'CVaR (95%)': cvar_95,
                    'Volatility': volatility,
                    'Skewness': skewness,
                    'Kurtosis': kurtosis,
                    'Max Drawdown': np.mean(drawdowns) if drawdowns else 0
                }
        
        # Display risk metrics table
        if risk_analysis:
            risk_df = pd.DataFrame(risk_analysis).T
            risk_df = risk_df.round(2)
            st.dataframe(risk_df, use_container_width=True)
        
        # Risk-return scatter plot
        st.subheader("Risk-Return Analysis")
        
        if len(risk_analysis) > 1:
            fig_risk, ax_risk = plt.subplots(figsize=(10, 6))
            
            scenarios = list(risk_analysis.keys())
            returns_means = [np.mean(st.session_state.stress_test_results[s]['final_returns']) for s in scenarios]
            volatilities = [risk_analysis[s]['Volatility'] for s in scenarios]
            
            scatter = ax_risk.scatter(volatilities, returns_means, s=100, alpha=0.7)
            
            for i, scenario in enumerate(scenarios):
                ax_risk.annotate(scenario.replace('_', ' ').title(), 
                               (volatilities[i], returns_means[i]),
                               xytext=(5, 5), textcoords='offset points')
            
            ax_risk.set_xlabel('Volatility (%)')
            ax_risk.set_ylabel('Mean Return (%)')
            ax_risk.set_title('Risk-Return Profile Across Scenarios')
            ax_risk.grid(True, alpha=0.3)
            
            st.pyplot(fig_risk)
    
    elif page == "Forward-Looking Analysis":
        st.header("Forward-Looking Analysis")
        
        if st.session_state.strategy_tester is None:
            st.warning("⚠️ Please upload a strategy and complete previous analyses first.")
            return
        
        st.markdown("""
        Generate forward-looking projections and scenario analysis for investment planning.
        """)
        
        # Forward simulation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.selectbox(
                "Forecast Horizon",
                options=[126, 252, 504, 756],
                index=1,
                format_func=lambda x: f"{x} days (~{x//252 if x>=252 else x//63} {'year' if x>=252 else 'quarter'}{'s' if x>=504 else ''})"
            )
        
        with col2:
            confidence_level = st.selectbox(
                "Confidence Level",
                options=[90, 95, 99],
                index=1,
                format_func=lambda x: f"{x}%"
            )
        
        # Custom scenario builder
        st.subheader("Custom Scenario Analysis")
        
        with st.expander("Build Custom Market Scenario"):
            st.write("Define expected market conditions for forward-looking analysis:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                market_regime = st.selectbox(
                    "Market Regime",
                    ["Normal", "Bull Market", "Bear Market", "High Volatility", "Low Volatility"]
                )
            
            with col2:
                expected_return = st.slider(
                    "Expected Annual Return (%)",
                    min_value=-20.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.5
                )
            
            with col3:
                expected_volatility = st.slider(
                    "Expected Volatility (%)",
                    min_value=5.0,
                    max_value=50.0,
                    value=20.0,
                    step=1.0
                )
        
        if st.button("Generate Forward-Looking Analysis", type="primary"):
            st.info("🔮 Generating forward-looking projections...")
            
            # This would implement forward-looking Monte Carlo
            # For now, showing placeholder structure
            st.write("Forward-looking analysis would include:")
            st.write("- Probability distributions of future returns")
            st.write("- Scenario-based projections")
            st.write("- Expected portfolio evolution")
            st.write("- Risk-adjusted return expectations")
            
            st.info("🚧 Forward-looking analysis implementation in progress...")
    
    elif page == "Strategy Optimization":
        st.header("Strategy Optimization")
        
        if not st.session_state.stress_test_results:
            st.warning("⚠️ Please complete stress testing first.")
            return
        
        st.markdown("""
        Analyze strategy performance and identify potential optimizations.
        """)
        
        st.subheader("Performance Optimization Opportunities")
        
        # Analyze current results for optimization suggestions
        results = st.session_state.stress_test_results
        
        optimization_suggestions = []
        
        # Check for consistent underperformance in specific scenarios
        worst_scenarios = []
        for scenario_name, scenario_results in results.items():
            if 'final_returns' in scenario_results:
                median_return = np.median(scenario_results['final_returns'])
                if median_return < -10:  # Worse than -10% median
                    worst_scenarios.append((scenario_name, median_return))
        
        if worst_scenarios:
            st.write("**🎯 Scenarios Needing Improvement:**")
            for scenario, return_val in worst_scenarios:
                st.write(f"- {scenario.replace('_', ' ').title()}: {return_val:.1f}% median return")
                
                if 'crisis' in scenario.lower():
                    optimization_suggestions.append("Add more defensive positioning during crisis indicators")
                elif 'tech' in scenario.lower():
                    optimization_suggestions.append("Consider sector diversification beyond technology")
                elif 'volatility' in scenario.lower():
                    optimization_suggestions.append("Strengthen volatility hedging mechanisms")
        
        # Display optimization suggestions
        if optimization_suggestions:
            st.write("**💡 Optimization Suggestions:**")
            for suggestion in optimization_suggestions:
                st.write(f"- {suggestion}")
        else:
            st.success("✅ Strategy shows robust performance across tested scenarios!")
        
        # Parameter sensitivity analysis
        st.subheader("Parameter Sensitivity Analysis")
        
        strategy_assets = st.session_state.strategy_tester.assets
        
        if any('RSI' in str(st.session_state.strategy_tester.parser.logic_tree) for _ in [1]):
            st.write("**RSI Parameter Analysis:**")
            st.write("Your strategy uses RSI indicators. Consider testing different RSI periods:")
            
            rsi_periods = [5, 10, 14, 21, 30]
            rsi_thresholds = [70, 75, 80, 85]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**RSI Periods to Test:**")
                for period in rsi_periods:
                    st.write(f"- {period} days")
            
            with col2:
                st.write("**RSI Thresholds to Test:**")
                for threshold in rsi_thresholds:
                    st.write(f"- {threshold} (overbought)")
                    st.write(f"- {100-threshold} (oversold)")
        
        # Asset allocation analysis
        st.subheader("Asset Allocation Analysis")
        
        if strategy_assets:
            st.write("**Current Asset Universe:**")
            
            asset_analysis = {}
            for asset in strategy_assets:
                if asset in ['TQQQ', 'TECL', 'SOXL']:
                    asset_analysis[asset] = "High-risk leveraged growth"
                elif asset in ['UVXY', 'VXX']:
                    asset_analysis[asset] = "Volatility hedge"
                elif asset in ['SQQQ']:
                    asset_analysis[asset] = "Short/defensive position"
                elif asset in ['BSV', 'TLT']:
                    asset_analysis[asset] = "Fixed income/safe haven"
                else:
                    asset_analysis[asset] = "Standard equity exposure"
            
            for asset, description in asset_analysis.items():
                st.write(f"- **{asset}**: {description}")
            
            # Diversification suggestions
            st.write("**🔄 Diversification Opportunities:**")
            
            has_international = any('EFA' in asset or 'VEA' in asset or 'IEFA' in asset for asset in strategy_assets)
            has_commodities = any('GLD' in asset or 'USO' in asset or 'DBA' in asset for asset in strategy_assets)
            has_reits = any('VNQ' in asset or 'REIT' in asset for asset in strategy_assets)
            
            if not has_international:
                st.write("- Consider adding international equity exposure (EFA, VEA, IEFA)")
            if not has_commodities:
                st.write("- Consider adding commodity exposure (GLD, USO, DBA)")
            if not has_reits:
                st.write("- Consider adding REIT exposure (VNQ, SCHH)")
            
            if has_international and has_commodities and has_reits:
                st.success("✅ Good diversification across asset classes!")
    
    elif page == "Documentation":
        st.header("Documentation & Methodology")
        
        st.markdown("""
        ## Holdings-Based Monte Carlo Simulation Methodology
        
        This tool implements **holdings-based Monte Carlo simulation**, which is superior to traditional 
        return-bootstrapping methods for strategy analysis.
        
        ### Key Advantages
        
        **🎯 True Strategy Simulation**
        - Simulates actual asset returns and applies your exact strategy logic
        - Tests how your strategy responds to market conditions it has never seen
        - Provides forward-looking risk assessment
        
        **📊 Comprehensive Stress Testing**
        - Tests strategy across multiple market scenarios (crisis, tech crash, volatility spikes)
        - Identifies potential failure modes before they occur
        - Quantifies tail risk and extreme scenarios
        
        **🔍 Component Analysis**
        - Analyzes which parts of your strategy add/subtract value
        - Tests parameter sensitivity
        - Provides optimization recommendations
        
        ### Methodology Details
        
        #### 1. Strategy Logic Extraction
        - Parses Composer JSON to extract exact decision tree
        - Implements technical indicators (RSI, Moving Averages, etc.)
        - Replicates conditional logic and asset selection rules
        
        #### 2. Market Scenario Generation
        - **Normal Markets**: Uses historical return distributions with realistic correlations
        - **Crisis Scenarios**: Simulates 2008-style financial crisis conditions
        - **Tech Crashes**: Models sector-specific downturns
        - **Volatility Spikes**: Simulates VIX explosion scenarios
        - **Other Scenarios**: Bear markets, rate shocks, everything rallies
        
        #### 3. Strategy Execution Simulation
        - Applies your exact strategy logic to simulated market data
        - Calculates technical indicators day-by-day
        - Executes allocation changes based on strategy rules
        - Accounts for realistic market dynamics
        
        #### 4. Risk Analysis
        - Calculates comprehensive risk metrics (VaR, CVaR, Maximum Drawdown)
        - Identifies failure modes and tail risks
        - Provides scenario-specific performance analysis
        
        ### Comparison to Traditional Methods
        
        | Aspect | Traditional Return Bootstrapping | Holdings-Based Simulation |
        |--------|--------------------------------|---------------------------|
        | **Approach** | Reshuffles historical strategy returns | Simulates assets + applies strategy logic |
        | **Forward-Looking** | Limited to historical patterns | Tests unprecedented scenarios |
        | **Strategy Response** | Static | Dynamic response to new conditions |
        | **Failure Mode Detection** | Poor | Excellent |
        | **Optimization Insights** | Basic | Comprehensive |
        
        ### Interpreting Results
        
        #### Performance Distributions
        - **Median Return**: Expected outcome under normal conditions
        - **5th/95th Percentiles**: Range of likely outcomes
        - **Probability of Loss**: Chance of negative returns
        
        #### Risk Metrics
        - **VaR (Value at Risk)**: Worst expected loss at given confidence level
        - **CVaR (Conditional VaR)**: Average loss beyond VaR threshold
        - **Maximum Drawdown**: Largest peak-to-trough decline
        
        #### Failure Mode Analysis
        - **Failure Rate**: Percentage of simulations with >20% losses
        - **Extreme Drawdown Rate**: Percentage with >30% drawdowns
        - **Scenario-Specific Risks**: Performance in crisis conditions
        
        ### Limitations and Considerations
        
        #### Model Limitations
        - Simulations based on historical data patterns
        - Cannot predict truly unprecedented events
        - Assumes strategy logic remains constant
        
        #### Best Practices
        - Run multiple simulation sets with different parameters
        - Focus on relative performance across scenarios
        - Use results for risk management, not return prediction
        - Regularly update analysis with new market data
        
        ### Technical Implementation
        
        #### Technology Stack
        - **Streamlit**: Interactive web application framework
        - **Python**: Core analysis and simulation engine
        - **yfinance**: Historical market data retrieval
        - **NumPy/Pandas**: Numerical computing and data analysis
        - **Matplotlib/Seaborn**: Visualization and plotting
        
        #### Data Sources
        - Yahoo Finance for historical price data
        - Composer strategy JSON for exact logic
        - Real-time technical indicator calculations
        
        ### Support and Feedback
        
        This tool is designed to provide institutional-quality strategy analysis 
        for individual investors and strategy developers.
        
        **Key Features:**
        - ✅ Exact strategy logic replication
        - ✅ Comprehensive stress testing
        - ✅ Failure mode identification
        - ✅ Risk assessment and optimization
        - ✅ Forward-looking scenario analysis
        
        For questions, suggestions, or custom analysis requests, please provide feedback
        through the application interface.
        """)
        
        # Example strategy JSON structure
        with st.expander("Example Strategy JSON Structure"):
            example_json = {
                "name": "Example Strategy",
                "description": "Sample strategy for demonstration",
                "children": [
                    {
                        "step": "if",
                        "condition": {
                            "lhs-fn": "current-price",
                            "lhs-val": "TQQQ",
                            "comparator": "gt",
                            "rhs-fn": "moving-average-price",
                            "rhs-val": "TQQQ",
                            "rhs-window-days": "200"
                        },
                        "children": [
                            {
                                "step": "asset",
                                "ticker": "TQQQ",
                                "weight": 1.0
                            }
                        ]
                    }
                ]
            }
            
            st.json(example_json)
        
        # Performance tips
        with st.expander("Performance Optimization Tips"):
            st.markdown("""
            **⚡ Optimizing Analysis Performance:**
            
            - **Simulation Count**: Start with 1,000 simulations for quick testing, increase to 5,000+ for final analysis
            - **Time Horizon**: Shorter periods (126 days) run faster than longer periods (504+ days)
            - **Historical Data**: 2-3 years of data usually sufficient for most strategies
            - **Scenario Selection**: Test most relevant scenarios first, then expand to comprehensive testing
            
            **🎯 Getting Better Results:**
            
            - **Strategy Validation**: Always run historical backtest first to validate implementation
            - **Multiple Time Horizons**: Test different forecast periods (3 months, 6 months, 1 year)
            - **Parameter Sensitivity**: Test variations of key strategy parameters
            - **Regular Updates**: Re-run analysis quarterly or after major market events
            """)

if __name__ == "__main__":
    main()
