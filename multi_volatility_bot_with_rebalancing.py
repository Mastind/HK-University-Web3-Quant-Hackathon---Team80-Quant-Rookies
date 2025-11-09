"""
🤖 ROOSTOO 24/7 AUTO-TRADING BOT - 6 Crypto Version
Analyzes BTC, ETH, SOL, BNB, XRP, ADA with individual GARCH models
Each crypto has its own historical data and volatility model
"""

import pandas as pd
import numpy as np
import time
import requests
import hmac
import hashlib
import math
from datetime import datetime, timedelta
import warnings
import traceback
import sys
warnings.filterwarnings('ignore')

# =============================================================================
# ROOSTOO API INTEGRATION
# =============================================================================

class RoostooAPI:
    """Roostoo Exchange API Client"""
    
    def __init__(self, symbol='BTC/USD'):
        self.BASE_URL = "https://mock-api.roostoo.com"
        self.API_KEY = "olDFYeqSIQgk5phJgY4lWKNNaGpJiWnnecDeIWitZLV23DtnP15lGJJbKi1BsOPD"
        self.SECRET_KEY = "Azfl3pdnzckPEYVR3pHJf3CzPtESK3bjkOUKlwqZZOlEThheiqqlZNlPTPg1sRYM"
        
        self.symbol = symbol
        self.base_currency = symbol.split('/')[0]
        self.quote_currency = symbol.split('/')[1]
        
        # Trading rules for the 6 supported cryptocurrencies
        self.trading_rules = {
            'BTC': {'decimals': 5, 'commission_rate': 0.001, 'min_order': 0.00001},
            'ETH': {'decimals': 4, 'commission_rate': 0.001, 'min_order': 0.0001},
            'SOL': {'decimals': 3, 'commission_rate': 0.001, 'min_order': 0.001},
            'BNB': {'decimals': 3, 'commission_rate': 0.001, 'min_order': 0.0001},
            'XRP': {'decimals': 1, 'commission_rate': 0.001, 'min_order': 0.1},
            'ADA': {'decimals': 1, 'commission_rate': 0.001, 'min_order': 0.1},
        }
        
        rules = self.trading_rules.get(
            self.base_currency,
            {'decimals': 4, 'commission_rate': 0.001, 'min_order': 0.01}
        )
        
        self.DECIMALS = rules['decimals']
        self.COMMISSION_RATE = rules['commission_rate']
        self.MIN_ORDER_QUANTITY = rules['min_order']
        
    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _get_signed_headers(self, payload: dict = {}):
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        signature = hmac.new(
            self.SECRET_KEY.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'RST-API-KEY': self.API_KEY,
            'MSG-SIGNATURE': signature
        }

        return headers, payload, total_params
    
    def _round_quantity(self, quantity):
        """Round quantity to correct decimal places"""
        multiplier = 10 ** self.DECIMALS
        rounded = math.floor(quantity * multiplier) / multiplier
        return rounded
    
    def _validate_quantity(self, quantity):
        """Validate quantity meets minimum order requirements"""
        if quantity < self.MIN_ORDER_QUANTITY:
            return False
        return True

    def get_ticker(self, pair=None):
        url = f"{self.BASE_URL}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
        if pair:
            params['pair'] = pair
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            if data.get('Success') and 'Data' in data:
                if pair and pair in data['Data']:
                    return data['Data'][pair]
                else:
                    return data
            return data
        except:
            return None

    def get_balance(self):
        url = f"{self.BASE_URL}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            res = requests.get(url, headers=headers, params=payload, timeout=10)
            res.raise_for_status()
            return res.json()
        except:
            return None

    def place_order(self, side, quantity, price=None, order_type=None):
        """Place order with proper decimal handling"""
        url = f"{self.BASE_URL}/v3/place_order"
        
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        if order_type == 'LIMIT' and price is None:
            return None

        # Round quantity
        quantity = self._round_quantity(quantity)
        
        # Validate minimum
        if not self._validate_quantity(quantity):
            return None

        payload = {
            'pair': self.symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        if order_type == 'LIMIT':
            payload['price'] = str(price)

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        try:
            res = requests.post(url, headers=headers, data=total_params, timeout=10)
            res.raise_for_status()
            return res.json()
        except:
            return None

# =============================================================================
# HORUS API DATA FETCHER
# =============================================================================

class HorusDataFetcher:
    """Fetch historical OHLCV data from Horus API for 6 supported cryptos"""
    
    def __init__(self, api_key=None):
        self.base_url = "https://api-horus.com"
        self.api_key = api_key
        
        # Symbol mapping for the 6 supported cryptocurrencies
        self.symbol_map = {
            'BTCUSDT': 'BTC',
            'ETHUSDT': 'ETH',
            'SOLUSDT': 'SOL',
            'BNBUSDT': 'BNB',
            'XRPUSDT': 'XRP',
            'ADAUSDT': 'ADA',
        }
        
        self.interval_map = {
            '1m': '1d', '3m': '1d', '5m': '1d',
            '15m': '15m', '30m': '1h', '1h': '1h',
            '2h': '1h', '4h': '1h', '1d': '1d'
        }
    
    def _get_headers(self):
        headers = {}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        return headers
        
    def fetch_ohlcv(self, symbol='BTCUSDT', interval='15m', days=14):
        """
        Fetch historical price data for a specific cryptocurrency
        
        Args:
            symbol: e.g., 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'
            interval: '15m', '1h', '1d'
            days: Number of days of historical data
        
        Returns:
            DataFrame with OHLCV data for the specific crypto
        """
        
        # Map symbol to Horus format
        horus_symbol = self.symbol_map.get(symbol)
        
        if not horus_symbol:
            print(f"❌ {symbol} not supported. Supported: {list(self.symbol_map.keys())}")
            return None
        
        horus_interval = self.interval_map.get(interval, '1h')
        
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        params = {
            'asset': horus_symbol,
            'interval': horus_interval,
            'start': start_time,
            'end': end_time,
            'format': 'json'
        }
        
        try:
            url = f"{self.base_url}/market/price"
            headers = self._get_headers()
            
            print(f"   Fetching {horus_symbol} from Horus API...")
            print(f"   Time range: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d')}")
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or len(data) == 0:
                print(f"   ❌ No data returned for {horus_symbol}")
                return None
            
            print(f"   ✅ Received {len(data)} price points for {horus_symbol}")
            
            df = self._convert_to_ohlcv(data, horus_interval, horus_symbol)
            return df
            
        except Exception as e:
            print(f"   ❌ Error fetching {horus_symbol}: {e}")
            return None
    
    def _convert_to_ohlcv(self, price_data, interval, symbol):
        """Convert Horus price data to OHLCV format"""
        
        df = pd.DataFrame(price_data)
        
        if df.empty or 'timestamp' not in df.columns or 'price' not in df.columns:
            return None
        
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(inplace=True)
        
        if len(df) == 0:
            return None
        
        freq_map = {'15m': '15T', '1h': '1H', '1d': '1D'}
        freq = freq_map.get(interval, '1H')
        
        ohlcv = df['price'].resample(freq).agg(
            Open='first',
            High='max',
            Low='min',
            Close='last',
            Count='count'
        )
        
        ohlcv['Volume'] = ohlcv['Count'] * 100.0
        ohlcv = ohlcv.drop('Count', axis=1)
        ohlcv.dropna(inplace=True)
        
        if len(ohlcv) == 0:
            return None
        
        print(f"   ✅ Created {len(ohlcv)} OHLCV candles for {symbol}")
        print(f"   Price range: ${ohlcv['Low'].min():,.2f} - ${ohlcv['High'].max():,.2f}")
        
        return ohlcv

# =============================================================================
# GARCH VOLATILITY MODEL
# =============================================================================

class GARCHVolatility:
    """Calculate volatility using GARCH(p,q) for a specific cryptocurrency"""
    
    def __init__(self, crypto_name, p=2, q=3):
        self.crypto_name = crypto_name
        self.p = p
        self.q = q
        self.last_fitted = None
    
    def fit_predict(self, returns, lookback=50):
        """Fit GARCH model on returns for this specific crypto"""
        
        volatility = pd.Series(index=returns.index, dtype=float)
        
        for i in range(lookback, len(returns)):
            window = returns.iloc[i-lookback:i] * 100
            
            try:
                from arch import arch_model
                model = arch_model(window, vol='Garch', p=self.p, q=self.q, rescale=False)
                result = model.fit(disp='off', show_warning=False)
                forecast = result.forecast(horizon=1)
                volatility.iloc[i] = np.sqrt(forecast.variance.values[-1, 0]) / 100
            except:
                volatility.iloc[i] = window.std() / 100
        
        self.last_fitted = datetime.now()
        return volatility

# =============================================================================
# PORTFOLIO REBALANCER
# =============================================================================

class PortfolioRebalancer:
    """Automatically rebalance portfolio to target allocations with rate limiting and retry logic"""
    
    def __init__(self):
        # Target portfolio allocations (must sum to 100%)
        self.target_allocations = {
            'BTC': 0.30,   # 30%
            'ETH': 0.25,   # 25%
            'SOL': 0.20,   # 20%
            'ADA': 0.10,   # 10%
            'BNB': 0.10,   # 10%
            'XRP': 0.05,   # 5%
        }
        
        # Rebalancing threshold (1% deviation triggers rebalance)
        self.rebalance_threshold = 0.01
        
        self.last_rebalance = None
        self.last_trade_time = None
        
    def wait_for_rate_limit(self):
        """Wait if needed to respect 1 trade per minute limit"""
        if self.last_trade_time is not None:
            seconds_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds()
            if seconds_since_last_trade < 60:
                wait_time = 60 - seconds_since_last_trade
                print(f"   ⏳ Rate limit: Waiting {wait_time:.1f} seconds before next trade...")
                time.sleep(wait_time)
        
        self.last_trade_time = datetime.now()
    
    def execute_trade_with_retry(self, api, crypto, side, quantity, price, reason, trade_logger, max_retries=2):
        """Execute a trade with retry logic for failed attempts"""
        
        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries} in 15 seconds...")
                time.sleep(15)
                # Update price for retry
                ticker = api.get_ticker(f"{crypto}/USD")
                if ticker and 'LastPrice' in ticker:
                    price = float(ticker['LastPrice'])
                    print(f"   📊 Updated price for retry: ${price:,.2f}")
            
            print(f"   💸 {side}ing {crypto}: {quantity:.6f} @ ${price:,.2f}")
            
            order = api.place_order(
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            if order and order.get('Success'):
                order_detail = order.get('OrderDetail', {})
                filled_quantity = order_detail.get('FilledQuantity', quantity)
                filled_price = order_detail.get('FilledAverPrice', price)
                
                trade_logger.log_trade(
                    crypto=crypto,
                    action=side,
                    quantity=filled_quantity,
                    price=filled_price,
                    reason=reason,
                    order_id=order_detail.get('OrderID'),
                    success=True
                )
                print(f"   ✅ Trade successful on attempt {attempt + 1}")
                return True
            else:
                trade_logger.log_trade(
                    crypto=crypto,
                    action=side,
                    quantity=quantity,
                    price=price,
                    reason=reason,
                    success=False
                )
                print(f"   ❌ Trade failed on attempt {attempt + 1}")
                
                if attempt < max_retries:
                    print(f"   ⏳ Waiting 15 seconds before retry...")
                else:
                    print(f"   💥 All retry attempts failed for {crypto} {side}")
        
        return False
        
    def calculate_current_allocations(self, roostoo_apis):
        """Calculate current portfolio allocations across all cryptos"""
        
        print("\n💰 Calculating current portfolio allocations...")
        
        total_value = 0
        crypto_values = {}
        current_allocations = {}
        usd_balance = 0  # Initialize usd_balance to 0
        
        # Initialize all crypto allocations to 0
        for crypto in roostoo_apis.keys():
            crypto_values[crypto] = 0
            current_allocations[crypto] = 0
        
        # Get balances and current prices for all cryptos
        for crypto, api in roostoo_apis.items():
            try:
                # Get balance
                balance_data = api.get_balance()
                if not balance_data or not balance_data.get('Success'):
                    print(f"   ❌ Could not fetch balance for {crypto}")
                    continue
                
                wallet = balance_data.get('SpotWallet', {})
                
                # Get crypto balance
                crypto_balance = wallet.get(crypto, {}).get('Free', 0)
                if crypto_balance <= 0:
                    # No holdings for this crypto, keep at 0
                    continue
                
                # Get current price
                ticker = api.get_ticker(f"{crypto}/USD")
                if not ticker or 'LastPrice' not in ticker:
                    print(f"   ❌ Could not fetch price for {crypto}")
                    continue
                
                current_price = float(ticker['LastPrice'])
                crypto_value = crypto_balance * current_price
                crypto_values[crypto] = crypto_value
                total_value += crypto_value
                
                print(f"   {crypto}: {crypto_balance:.6f} × ${current_price:,.2f} = ${crypto_value:,.2f}")
                
            except Exception as e:
                print(f"   ❌ Error calculating {crypto} value: {e}")
                # Keep value at 0
        
        # Get USD balance
        try:
            usd_balance_data = roostoo_apis['BTC'].get_balance()  # Use any API for USD balance
            if usd_balance_data and usd_balance_data.get('Success'):
                wallet = usd_balance_data.get('SpotWallet', {})
                usd_balance = wallet.get('USD', {}).get('Free', 0)
                total_value += usd_balance
                print(f"   USD: ${usd_balance:,.2f}")
        except Exception as e:
            print(f"   ❌ Error fetching USD balance: {e}")
            usd_balance = 0
        
        # Calculate current allocations
        if total_value > 0:
            for crypto in roostoo_apis.keys():
                crypto_value = crypto_values.get(crypto, 0)
                current_allocations[crypto] = crypto_value / total_value
            
            current_allocations['USD'] = usd_balance / total_value
            
            print(f"\n   Total Portfolio Value: ${total_value:,.2f}")
            print(f"   Current Allocations:")
            for crypto, alloc in current_allocations.items():
                print(f"     {crypto}: {alloc:.1%}")
        else:
            print(f"\n   💰 Total Portfolio Value: $0.00")
            print(f"   Current Allocations: 100% USD (no crypto holdings)")
            # All in USD when no value
            for crypto in roostoo_apis.keys():
                current_allocations[crypto] = 0
            current_allocations['USD'] = 1.0  # 100% USD
        
        return current_allocations, total_value, crypto_values
        
    def check_rebalance_needed(self, current_allocations, total_value):
        """Check if rebalancing is needed based on threshold"""
        
        print("\n🔍 Checking if rebalancing is needed...")
        
        max_deviation = 0
        needs_rebalance = False
        rebalance_actions = {}
        
        # If portfolio is empty or very small, don't rebalance
        if total_value < 10:  # Less than $10 total
            print(f"   ⏸️  Portfolio too small (${total_value:.2f}), skipping rebalance")
            return False, {}
        
        for crypto, target_alloc in self.target_allocations.items():
            current_alloc = current_allocations.get(crypto, 0)
            deviation = abs(current_alloc - target_alloc)
            
            print(f"   {crypto}: Current {current_alloc:.1%} vs Target {target_alloc:.1%} | Deviation: {deviation:.1%}")
            
            if deviation > max_deviation:
                max_deviation = deviation
            
            if deviation > self.rebalance_threshold:
                needs_rebalance = True
                
                # Determine action needed
                if current_alloc > target_alloc:
                    action = 'SELL'
                    amount_pct = current_alloc - target_alloc
                else:
                    action = 'BUY' 
                    amount_pct = target_alloc - current_alloc
                
                rebalance_actions[crypto] = {
                    'action': action,
                    'amount_pct': amount_pct,
                    'current_alloc': current_alloc,
                    'target_alloc': target_alloc,
                    'deviation': deviation  # For prioritization
                }
        
        if needs_rebalance:
            print(f"\n   ⚠️  Rebalancing needed! Max deviation: {max_deviation:.1%}")
            # Sort by deviation (highest first) to prioritize largest imbalances
            sorted_actions = dict(sorted(
                rebalance_actions.items(), 
                key=lambda x: x[1]['deviation'], 
                reverse=True
            ))
            for crypto, action_info in sorted_actions.items():
                print(f"     {crypto}: {action_info['action']} {action_info['amount_pct']:.1%}")
            return needs_rebalance, sorted_actions
        else:
            print(f"\n   ✅ Portfolio within target ranges (max deviation: {max_deviation:.1%})")
            return needs_rebalance, {}
            
    def execute_rebalance(self, roostoo_apis, total_portfolio_value, current_allocations, rebalance_actions, trade_logger):
        """Execute the rebalancing trades with proper decimal handling, rate limiting, and retry logic"""
        
        print("\n🔄 Executing portfolio rebalance with 1 trade/minute limit and retry logic...")
        
        rebalance_count = 0
        successful_trades = 0
        total_actions = len([a for a in rebalance_actions.values() if a['action'] == 'SELL']) + \
                       len([a for a in rebalance_actions.values() if a['action'] == 'BUY'])
        
        print(f"   📊 Planned rebalance: {total_actions} trades needed")
        print(f"   ⏰ Estimated time: {total_actions} minutes")
        print(f"   🔄 Retry logic: 2 retries with 15-second delays")
        
        # First pass: Execute SELL orders to free up USD (prioritize largest deviations)
        for crypto, action_info in rebalance_actions.items():
            if action_info['action'] == 'SELL':
                # Wait for rate limit before each trade
                self.wait_for_rate_limit()
                
                api = roostoo_apis[crypto]
                
                # Calculate USD amount to sell
                usd_amount = total_portfolio_value * action_info['amount_pct']
                
                # Get current price
                ticker = api.get_ticker(f"{crypto}/USD")
                if not ticker or 'LastPrice' not in ticker:
                    print(f"   ❌ Could not get price for {crypto}, skipping sell")
                    continue
                
                current_price = float(ticker['LastPrice'])
                quantity = usd_amount / current_price
                
                # Round quantity using crypto-specific decimal places
                quantity = api._round_quantity(quantity)
                
                # Validate minimum order quantity
                if not api._validate_quantity(quantity):
                    print(f"   ⏸️  {crypto} sell quantity too small: {quantity}")
                    continue
                
                reason = f"Rebalance: {action_info['current_alloc']:.1%} → {action_info['target_alloc']:.1%}"
                
                # Execute sell order with retry logic
                success = self.execute_trade_with_retry(
                    api=api,
                    crypto=crypto,
                    side='SELL',
                    quantity=quantity,
                    price=current_price,
                    reason=reason,
                    trade_logger=trade_logger,
                    max_retries=2
                )
                
                if success:
                    successful_trades += 1
                rebalance_count += 1
        
        # Wait after sells before starting buys
        if rebalance_count > 0:
            print(f"   ⏳ Finished sell orders, waiting before buy orders...")
            self.wait_for_rate_limit()
        
        # Second pass: Execute BUY orders with available USD (prioritize largest deviations)
        for crypto, action_info in rebalance_actions.items():
            if action_info['action'] == 'BUY':
                # Wait for rate limit before each trade
                self.wait_for_rate_limit()
                
                api = roostoo_apis[crypto]
                
                # Calculate USD amount to buy
                usd_amount = total_portfolio_value * action_info['amount_pct']
                
                # Get current price
                ticker = api.get_ticker(f"{crypto}/USD")
                if not ticker or 'LastPrice' not in ticker:
                    print(f"   ❌ Could not get price for {crypto}, skipping buy")
                    continue
                
                current_price = float(ticker['LastPrice'])
                quantity = usd_amount / current_price
                
                # Round quantity using crypto-specific decimal places
                quantity = api._round_quantity(quantity)
                
                # Validate minimum order quantity
                if not api._validate_quantity(quantity):
                    print(f"   ⏸️  {crypto} buy quantity too small: {quantity}")
                    continue
                
                reason = f"Rebalance: {action_info['current_alloc']:.1%} → {action_info['target_alloc']:.1%}"
                
                # Execute buy order with retry logic
                success = self.execute_trade_with_retry(
                    api=api,
                    crypto=crypto,
                    side='BUY',
                    quantity=quantity,
                    price=current_price,
                    reason=reason,
                    trade_logger=trade_logger,
                    max_retries=2
                )
                
                if success:
                    successful_trades += 1
                rebalance_count += 1
        
        self.last_rebalance = datetime.now()
        
        if successful_trades > 0:
            print(f"\n   ✅ Rebalance completed: {successful_trades}/{rebalance_count} trades successful")
            print(f"   ⏰ Total time: {rebalance_count} minutes")
        else:
            print(f"\n   💥 Rebalance failed: 0/{rebalance_count} trades successful")
        
        return successful_trades

    def should_rebalance(self):
        """Check if it's appropriate to rebalance (not too frequent)"""
        if self.last_rebalance is None:
            return True
        
        hours_since_rebalance = (datetime.now() - self.last_rebalance).total_seconds() / 3600
        return hours_since_rebalance >= 1  # Rebalance at most every 1 hours
                        
# =============================================================================
# VOLATILITY STRATEGY
# =============================================================================

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class VolatilityStrategy:
    """Trade volatility expansions"""
    
    def __init__(self, vol_percentile, rsi_oversold, rsi_overbought, min_momentum):
        self.vol_percentile = vol_percentile
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_momentum = min_momentum
    
    def generate_signals(self, df):
        df['rsi'] = calculate_rsi(df['Close'])
        df['returns'] = df['Close'].pct_change()
        df['signal'] = -1
        df['confidence'] = 0.0
        
        vol_threshold = df['volatility'].quantile(self.vol_percentile)
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            if row['volatility'] < vol_threshold:
                continue
            
            if (row['rsi'] < self.rsi_oversold and row['returns'] > self.min_momentum):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'confidence'] = 0.75
            
            elif (row['rsi'] > self.rsi_overbought and row['returns'] < -self.min_momentum):
                df.loc[df.index[i], 'signal'] = 0
                df.loc[df.index[i], 'confidence'] = 0.75
            
            elif row['volatility'] > vol_threshold * 1.5:
                if row['returns'] > 0.005 and row['rsi'] > 55:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'confidence'] = 0.65
                
                elif row['returns'] < -0.005 and row['rsi'] < 45:
                    df.loc[df.index[i], 'signal'] = 0
                    df.loc[df.index[i], 'confidence'] = 0.65
        
        return df

# =============================================================================
# TRADE LOGGER
# =============================================================================

class TradeLogger:
    """Log all trading activity"""
    
    def __init__(self):
        self.trades = []
        self.start_time = datetime.now()
        
    def log_trade(self, crypto, action, quantity, price, reason, order_id=None, success=True):
        trade = {
            'timestamp': datetime.now(),
            'crypto': crypto,
            'action': action,
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'reason': reason,
            'order_id': order_id,
            'success': success
        }
        self.trades.append(trade)
        
        emoji = "✅" if success else "❌"
        print(f"{emoji} [{trade['timestamp'].strftime('%H:%M:%S')}] {action} {quantity:.6f} {crypto} @ ${price:,.2f}")
        if order_id:
            print(f"   Order ID: {order_id}")
    
    def get_summary(self):
        if not self.trades:
            return "No trades executed yet."
        
        total_trades = len(self.trades)
        successful = sum(1 for t in self.trades if t['success'])
        total_value = sum(t['value'] for t in self.trades if t['success'])
        
        by_crypto = {}
        for trade in self.trades:
            crypto = trade['crypto']
            if crypto not in by_crypto:
                by_crypto[crypto] = {'buy': 0, 'sell': 0, 'value': 0}
            
            if trade['success']:
                if trade['action'] == 'BUY':
                    by_crypto[crypto]['buy'] += 1
                else:
                    by_crypto[crypto]['sell'] += 1
                by_crypto[crypto]['value'] += trade['value']
        
        summary = f"\n{'='*80}\n"
        summary += f"📈 TRADING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Trades: {total_trades} (Success: {successful}, Failed: {total_trades - successful})\n"
        summary += f"Total Value Traded: ${total_value:,.2f}\n"
        summary += f"Uptime: {datetime.now() - self.start_time}\n\n"
        
        for crypto, stats in by_crypto.items():
            summary += f"{crypto}:\n"
            summary += f"  Buys: {stats['buy']}, Sells: {stats['sell']}\n"
            summary += f"  Total Value: ${stats['value']:,.2f}\n"
        
        summary += f"{'='*80}\n"
        
        return summary

# =============================================================================
# 24/7 AUTO-TRADING BOT WITH INDIVIDUAL CRYPTO MODELS
# =============================================================================

class AutoTradingBot:
    """
    Fully automated 24/7 trading bot for 6 cryptocurrencies
    Each crypto has its own GARCH model built from its own historical data
    """
    
    def __init__(self, horus_api_key, auto_trade=True, min_confidence=0.65):
        # Only the 6 supported cryptocurrencies
        self.cryptos = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA']
        
        self.auto_trade = auto_trade
        self.min_confidence = min_confidence
        
        self.config = {
            'interval': '15m',
            'lookback_days': 14,
            'garch_p': 2,
            'garch_q': 3,
            'vol_percentile': 0.65,
            'rsi_oversold': 40,
            'rsi_overbought': 60,
            'min_momentum': 0.001,
            'position_size_pct': 0.50,
            'max_position_usd': 30000,
            'min_position_usd': 50,
        }
        
        # Individual API instances for each crypto
        self.roostoo_apis = {
            crypto: RoostooAPI(f"{crypto}/USD") for crypto in self.cryptos
        }
        
        # Single data fetcher (handles all cryptos)
        self.data_fetcher = HorusDataFetcher(api_key=horus_api_key)
        
        # Individual GARCH models for each crypto (built from their own data)
        self.garch_models = {}
        
        # Shared strategy
        self.strategy = VolatilityStrategy(
            vol_percentile=self.config['vol_percentile'],
            rsi_oversold=self.config['rsi_oversold'],
            rsi_overbought=self.config['rsi_overbought'],
            min_momentum=self.config['min_momentum']
        )
        
        self.logger = TradeLogger()
        self.last_model_rebuild = None
        self.cycle_count = 0
        
        # Crypto-specific data cache
        self.crypto_data = {}
        
        # Build initial models for each crypto
        self.rebuild_models()
        
        self.rebalancer = PortfolioRebalancer()
    
    def rebuild_models(self):
        """Rebuild GARCH models for each cryptocurrency using its own historical data"""
        print("\n🔧 Rebuilding individual models for each cryptocurrency...")
        print("="*80)
        
        for crypto in self.cryptos:
            print(f"\n📊 Building model for {crypto}...")
            
            # Create individual GARCH model for this crypto
            self.garch_models[crypto] = GARCHVolatility(
                crypto_name=crypto,
                p=self.config['garch_p'],
                q=self.config['garch_q']
            )
            
            print(f"   ✅ {crypto} model initialized")
        
        self.last_model_rebuild = datetime.now()
        print(f"\n✅ All models rebuilt at {self.last_model_rebuild.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def should_rebuild_models(self):
        """Check if it's time to rebuild models (daily)"""
        if self.last_model_rebuild is None:
            return True
        
        hours_since_rebuild = (datetime.now() - self.last_model_rebuild).total_seconds() / 3600
        return hours_since_rebuild >= 24
    
    def analyze_crypto(self, crypto):
        """
        Analyze a single cryptocurrency using its own historical data and GARCH model
        
        This ensures each crypto has its own:
        - Historical price data from Horus
        - GARCH volatility model
        - Trading signals
        """
        symbol = f"{crypto}USDT"
        
        print(f"\n{'─'*80}")
        print(f"📊 Analyzing {crypto}")
        print(f"{'─'*80}")
        
        try:
            # Fetch SPECIFIC data for THIS crypto from Horus
            print(f"Fetching {crypto}-specific historical data from Horus...")
            df = self.data_fetcher.fetch_ohlcv(
                symbol=symbol,  # BTCUSDT, ETHUSDT, SOLUSDT, etc.
                interval=self.config['interval'],
                days=self.config['lookback_days']
            )
            
            if df is None or len(df) < 100:
                print(f"   ❌ Insufficient data for {crypto}")
                return None
            
            # Calculate GARCH volatility using THIS crypto's own model
            print(f"Calculating {crypto}-specific GARCH volatility...")
            df['volatility'] = self.garch_models[crypto].fit_predict(df['Close'].pct_change())
            df = df.dropna()
            
            if len(df) < 100:
                print(f"   ❌ Insufficient data after volatility calculation for {crypto}")
                return None
            
            print(f"   ✅ {crypto} volatility model fitted on {len(df)} candles")
            
            # Generate signals using THIS crypto's data
            df_with_signals = self.strategy.generate_signals(df)
            
            # Get latest signal
            latest_row = df_with_signals.iloc[-1]
            
            # Get Roostoo price for THIS crypto
            ticker = self.roostoo_apis[crypto].get_ticker(f"{crypto}/USD")
            roostoo_price = float(ticker['LastPrice']) if ticker and 'LastPrice' in ticker else latest_row['Close']
            
            analysis = {
                'crypto': crypto,
                'df': df_with_signals,
                'signal': latest_row['signal'],
                'confidence': latest_row['confidence'],
                'price': roostoo_price,
                'volatility': latest_row['volatility'],
                'rsi': latest_row['rsi'],
                'returns': latest_row['returns'],
                'data_points': len(df_with_signals)
            }
            
            # Cache the data
            self.crypto_data[crypto] = df_with_signals
            
            signal_text = "BUY 🟢" if analysis['signal'] == 1 else "SELL 🔴" if analysis['signal'] == 0 else "HOLD ➖"
            print(f"   Signal: {signal_text}")
            print(f"   Confidence: {analysis['confidence']:.0%}")
            print(f"   Price: ${analysis['price']:,.2f}")
            print(f"   RSI: {analysis['rsi']:.1f}")
            print(f"   Volatility: {analysis['volatility']:.4f}")
            print(f"   Data Points: {analysis['data_points']}")
            
            return analysis
            
        except Exception as e:
            print(f"   ❌ Error analyzing {crypto}: {e}")
            traceback.print_exc()
            return None
    
    def execute_trade(self, analysis):
        """Execute a trade based on analysis with rate limiting and retry logic"""
        crypto = analysis['crypto']
        signal = analysis['signal']
        confidence = analysis['confidence']
        price = analysis['price']
        
        if signal == -1:
            return False
        
        if confidence < self.min_confidence:
            print(f"⏸️  {crypto}: Confidence {confidence:.0%} < minimum {self.min_confidence:.0%}, skipping")
            return False
        
        # Wait for rate limit before executing trade
        if self.rebalancer.last_trade_time is not None:
            seconds_since_last_trade = (datetime.now() - self.rebalancer.last_trade_time).total_seconds()
            if seconds_since_last_trade < 60:
                wait_time = 60 - seconds_since_last_trade
                print(f"   ⏳ Rate limit: Waiting {wait_time:.1f} seconds before trading {crypto}...")
                time.sleep(wait_time)
        
        self.rebalancer.last_trade_time = datetime.now()
        
        balance = self.roostoo_apis[crypto].get_balance()
        if not balance or not balance.get('Success'):
            print(f"❌ {crypto}: Could not fetch balance")
            return False
        
        wallet = balance.get('SpotWallet', {})
        
        if signal == 1:  # BUY
            usd_balance = wallet.get('USD', {}).get('Free', 0)
            
            if usd_balance <= 0:
                print(f"⏸️  {crypto}: No USD balance available")
                return False
            
            position_usd = min(
                usd_balance * self.config['position_size_pct'],
                self.config['max_position_usd']
            )
            
            if position_usd < self.config['min_position_usd']:
                print(f"⏸️  {crypto}: Position ${position_usd:.2f} < minimum ${self.config['min_position_usd']}")
                return False
            
            quantity = position_usd / price
            side = 'BUY'
            reason = f"Vol signal (RSI: {analysis['rsi']:.1f}, Vol: {analysis['volatility']:.4f})"
            
        else:  # SELL
            crypto_balance = wallet.get(crypto, {}).get('Free', 0)
            
            if crypto_balance <= 0:
                print(f"⏸️  {crypto}: No {crypto} balance to sell")
                return False
            
            quantity = crypto_balance * self.config['position_size_pct']
            side = 'SELL'
            reason = f"Vol signal (RSI: {analysis['rsi']:.1f}, Vol: {analysis['volatility']:.4f})"
        
        print(f"\n💸 Executing {side} order for {crypto}...")
        print(f"   Quantity: {quantity:.6f}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Confidence: {confidence:.0%}")
        
        # Use the retry logic for volatility trades too
        api = self.roostoo_apis[crypto]
        
        # Round quantity
        quantity = api._round_quantity(quantity)
        
        # Validate minimum order quantity
        if not api._validate_quantity(quantity):
            print(f"⏸️  {crypto}: Quantity {quantity} below minimum")
            return False
        
        success = self.rebalancer.execute_trade_with_retry(
            api=api,
            crypto=crypto,
            side=side,
            quantity=quantity,
            price=price,
            reason=reason,
            trade_logger=self.logger,
            max_retries=2
        )
        
        return success
                
    def run_cycle(self):
        """Run a single trading cycle"""
        self.cycle_count += 1
        
        print("\n" + "="*80)
        print(f"🔄 CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"Analyzing 6 cryptocurrencies with individual GARCH models...")
        
        if self.should_rebuild_models():
            self.rebuild_models()
        
        # =========================================================================
        # PORTFOLIO REBALANCING CHECK (AUTOMATIC)
        # =========================================================================
        if self.auto_trade and self.rebalancer.should_rebalance():
            print(f"\n💰 Checking portfolio rebalancing...")
            
            try:
                # Calculate current allocations
                current_allocations, total_value, crypto_values = self.rebalancer.calculate_current_allocations(
                    self.roostoo_apis
                )
                
                if total_value > 100:  # Only rebalance if we have meaningful portfolio (>$100)
                    # Check if rebalancing is needed
                    needs_rebalance, rebalance_actions = self.rebalancer.check_rebalance_needed(
                        current_allocations, total_value
                    )
                    
                    # Execute rebalance automatically if needed
                    if needs_rebalance and rebalance_actions:
                        print(f"\n🎯 Target Allocations:")
                        for crypto, target in self.rebalancer.target_allocations.items():
                            print(f"   {crypto}: {target:.1%}")
                        
                        total_trades_needed = len([a for a in rebalance_actions.values() if a['action'] == 'SELL']) + \
                                             len([a for a in rebalance_actions.values() if a['action'] == 'BUY'])
                        
                        if total_trades_needed > 3:  # If too many trades needed, warn user
                            print(f"\n⚠️  WARNING: Rebalancing requires {total_trades_needed} trades ({total_trades_needed} minutes)")
                            print(f"   This will significantly impact trading cycles")
                        
                        print(f"\n🔄 AUTOMATIC REBALANCING IN PROGRESS...")
                        rebalance_count = self.rebalancer.execute_rebalance(
                            self.roostoo_apis,
                            total_value,
                            current_allocations,
                            rebalance_actions,
                            self.logger
                        )
                        if rebalance_count > 0:
                            print(f"   ✅ Portfolio automatically rebalanced with {rebalance_count} trades")
                            # Skip volatility trading this cycle since we used our trades for rebalancing
                            print(f"\n✅ Cycle #{self.cycle_count} completed (rebalancing only)")
                            return
                        else:
                            print(f"   ⏸️  Rebalancing failed or no trades executed")
            except Exception as e:
                print(f"   ❌ Error during rebalancing check: {e}")
                traceback.print_exc()
        
        # =========================================================================
        # VOLATILITY STRATEGY ANALYSIS (existing code)
        # =========================================================================
        print(f"\n📊 Running volatility strategy analysis...")
        analyses = {}
        for crypto in self.cryptos:
            analysis = self.analyze_crypto(crypto)
            if analysis:
                analyses[crypto] = analysis
        
        # Execute trades if auto-trading is enabled
        if self.auto_trade:
            print(f"\n💼 Auto-trading enabled - executing high-confidence trades...")
            trade_count = 0
            for crypto, analysis in analyses.items():
                if analysis['signal'] != -1 and analysis['confidence'] >= self.min_confidence:
                    if self.execute_trade(analysis):
                        trade_count += 1
                    # Don't sleep here anymore - rate limiting is handled in execute_trade
            
            if trade_count == 0:
                print("   ⏸️  No trades executed this cycle")
        else:
            print(f"\n⏸️  Auto-trading disabled - showing recommendations only")
            for crypto, analysis in analyses.items():
                if analysis['signal'] != -1 and analysis['confidence'] >= self.min_confidence:
                    action = "BUY" if analysis['signal'] == 1 else "SELL"
                    print(f"   💡 {action} {crypto} @ ${analysis['price']:,.2f} (confidence: {analysis['confidence']:.0%})")
        
        print(f"\n✅ Cycle #{self.cycle_count} completed")
    
    def run_forever(self, cycle_interval_minutes=5):
        """Run bot forever"""
        print("\n" + "🚀"*40)
        print("24/7 AUTO-TRADING BOT STARTED - 6 CRYPTO VERSION")
        print("🚀"*40)
        print(f"\n⚙️  Configuration:")
        print(f"   Cryptos: {', '.join(self.cryptos)}")
        print(f"   Each crypto has its own:")
        print(f"     - Historical data from Horus API")
        print(f"     - Individual GARCH volatility model")
        print(f"     - Separate trading signals")
        print(f"   Cycle Interval: {cycle_interval_minutes} minutes")
        print(f"   Auto-trading: {'ENABLED ✅' if self.auto_trade else 'DISABLED ⏸️'}")
        print(f"   Min Confidence: {self.min_confidence:.0%}")
        print(f"   Position Size: {self.config['position_size_pct']:.0%} of balance")
        print(f"   Max Position: ${self.config['max_position_usd']:,.0f}")
        print(f"   Model Rebuild: Every 24 hours")
        print("\n⚠️  Press Ctrl+C to stop\n")
        print("="*80)
        
        cycle_interval_seconds = cycle_interval_minutes * 60
        
        try:
            while True:
                try:
                    self.run_cycle()
                    
                    if self.cycle_count % 10 == 0:
                        print(self.logger.get_summary())
                    
                    next_cycle = datetime.now() + timedelta(seconds=cycle_interval_seconds)
                    print(f"\n⏰ Next cycle at {next_cycle.strftime('%H:%M:%S')}")
                    print(f"   Sleeping for {cycle_interval_minutes} minutes...")
                    
                    time.sleep(cycle_interval_seconds)
                    
                except Exception as e:
                    print(f"\n❌ Error in cycle: {e}")
                    print(f"   Traceback: {traceback.format_exc()}")
                    print(f"   Waiting 60 seconds before retry...")
                    time.sleep(60)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutdown requested by user")
            print(self.logger.get_summary())
            print("\n👋 Bot stopped gracefully")
            sys.exit(0)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point"""
    print("="*80)
    print("🤖 ROOSTOO 24/7 AUTO-TRADING BOT - 6 CRYPTO VERSION")
    print("="*80)
    print("Cryptocurrencies: BTC, ETH, SOL, BNB, XRP, ADA")
    print("\nEach cryptocurrency has:")
    print("  • Its own historical data from Horus API")
    print("  • Individual GARCH volatility model")
    print("  • Separate trading signals and analysis")
    print("="*80)
    
    HORUS_API_KEY = "2458b11dad77377f2c13f1401776436cd9f386148220635cf399816355bc28c2"
    
    print("\n⚙️  CONFIGURATION:")
    print("="*80)
    
    auto_trade_input = input("\nEnable AUTO-TRADING? (yes/no) [no]: ").strip().lower()
    auto_trade = auto_trade_input in ['yes', 'y']
    
    if auto_trade:
        print("\n⚠️  WARNING: Auto-trading is ENABLED!")
        print("   The bot will automatically execute trades on your Roostoo account.")
        confirm = input("   Type 'CONFIRM' to proceed: ").strip()
        if confirm != 'CONFIRM':
            print("   Auto-trading disabled for safety.")
            auto_trade = False
    
    min_confidence_input = input("\nMinimum confidence for trades? (0.60-0.80) [0.65]: ").strip()
    try:
        min_confidence = float(min_confidence_input) if min_confidence_input else 0.65
        min_confidence = max(0.60, min(0.80, min_confidence))
    except:
        min_confidence = 0.65
    
    cycle_minutes_input = input("\nCycle interval in minutes? (1-60) [5]: ").strip()
    try:
        cycle_minutes = int(cycle_minutes_input) if cycle_minutes_input else 5
        cycle_minutes = max(1, min(60, cycle_minutes))
    except:
        cycle_minutes = 5
    
    max_position_input = input("\nMaximum position size per trade in USD? [30000]: ").strip()
    try:
        max_position_usd = float(max_position_input) if max_position_input else 30000
        max_position_usd = max(50, max_position_usd)
    except:
        max_position_usd = 30000
    
    print("\n" + "="*80)
    print("FINAL CONFIGURATION:")
    print("="*80)
    print(f"Cryptocurrencies: BTC, ETH, SOL, BNB, XRP, ADA")
    print(f"Auto-trading: {'ENABLED ✅' if auto_trade else 'DISABLED ⏸️'}")
    print(f"Min Confidence: {min_confidence:.0%}")
    print(f"Cycle Interval: {cycle_minutes} minutes")
    print(f"Max Position: ${max_position_usd:,.0f}")
    print(f"Rebalance Threshold: 1% deviation")
    print(f"Target Allocations:")
    print(f"  BTC: 30%, ETH: 25%, SOL: 20%, ADA: 10%, BNB: 10%, XRP: 5%")
    print("="*80)
    
    if auto_trade:
        print("\n⚠️  FINAL WARNING: Real trades will be executed!")
        input("Press ENTER to start, or Ctrl+C to cancel...")
    
    bot = AutoTradingBot(
        horus_api_key=HORUS_API_KEY,
        auto_trade=auto_trade,
        min_confidence=min_confidence
    )
    
    bot.config['max_position_usd'] = max_position_usd
    
    bot.run_forever(cycle_interval_minutes=cycle_minutes)

if __name__ == '__main__':
    try:
        from arch import arch_model
        main()
    except ImportError:
        print("❌ Required package 'arch' not installed.")
        print("   Install with: pip install arch")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)