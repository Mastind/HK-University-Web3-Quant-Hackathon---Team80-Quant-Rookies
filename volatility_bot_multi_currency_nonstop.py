"""
🤖 ROOSTOO 24/7 AUTO-TRADING BOT - Multi-Crypto System
Automatically analyzes and trades BTC, ETH, SOL every 5 minutes
Rebuilds GARCH models daily for optimal performance
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
        
        self.trading_rules = {
            'BTC': {'decimals': 5, 'commission_rate': 0.001},
            'ETH': {'decimals': 4, 'commission_rate': 0.001},
            'SOL': {'decimals': 3, 'commission_rate': 0.001},
            'BNB': {'decimals': 3, 'commission_rate': 0.001, 'min_order': 0.0001},
            'XRP': {'decimals': 1, 'commission_rate': 0.001, 'min_order': 0.1},       # Ripple
            'ADA': {'decimals': 1, 'commission_rate': 0.001, 'min_order': 0.1} 
        }
        
        rules = self.trading_rules.get(self.base_currency, {'decimals': 5, 'commission_rate': 0.001})
        self.DECIMALS = rules['decimals']
        self.COMMISSION_RATE = rules['commission_rate']
        
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
        multiplier = 10 ** self.DECIMALS
        return math.floor(quantity * multiplier) / multiplier

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
        url = f"{self.BASE_URL}/v3/place_order"
        
        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        if order_type == 'LIMIT' and price is None:
            return None

        quantity = self._round_quantity(quantity)
        
        if quantity <= 0:
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
    """Fetch historical OHLCV data from Horus API"""
    
    def __init__(self, api_key=None):
        self.base_url = "https://api-horus.com"
        self.api_key = api_key
        
        self.symbol_map = {
            'BTCUSDT': 'BTC',
            'ETHUSDT': 'ETH',
            'SOLUSDT': 'SOL'
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
        horus_symbol = self.symbol_map.get(symbol, 'BTC')
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
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or len(data) == 0:
                return None
            
            df = self._convert_to_ohlcv(data, horus_interval)
            return df
            
        except:
            return None
    
    def _convert_to_ohlcv(self, price_data, interval):
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
        
        return ohlcv

# =============================================================================
# GARCH VOLATILITY MODEL
# =============================================================================

class GARCHVolatility:
    """Calculate volatility using GARCH(p,q)"""
    
    def __init__(self, p=2, q=3):
        self.p = p
        self.q = q
    
    def fit_predict(self, returns, lookback=50):
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
        
        return volatility

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
        print(f"{emoji} [{trade['timestamp'].strftime('%H:%M:%S')}] {action} {quantity:.6f} {crypto} @ ${price:,.2f} - {reason}")
        if order_id:
            print(f"   Order ID: {order_id}")
    
    def log_cycle(self, cycle_num, analyses):
        print(f"\n📊 Cycle #{cycle_num} Summary:")
        for crypto, analysis in analyses.items():
            if hasattr(analysis, 'signal'):
                signal_text = "BUY" if analysis.signal == 1 else "SELL" if analysis.signal == 0 else "HOLD"
                print(f"   {crypto}: {signal_text} (confidence: {analysis.confidence:.0%})")
    
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
# 24/7 AUTO-TRADING BOT
# =============================================================================

class AutoTradingBot:
    """Fully automated 24/7 trading bot"""
    
    def __init__(self, horus_api_key, auto_trade=True, min_confidence=0.65):
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
            'position_size_pct': 0.60,  # Reduced to 60% for safety
            'max_position_usd': 30000,   # Maximum $30000 per trade
            'min_position_usd': 50
        }
        
        self.roostoo_apis = {
            crypto: RoostooAPI(f"{crypto}/USD") for crypto in self.cryptos
        }
        
        self.data_fetcher = HorusDataFetcher(api_key=horus_api_key)
        self.garch_model = None
        self.strategy = None
        
        self.logger = TradeLogger()
        self.last_model_rebuild = None
        self.cycle_count = 0
        
        # Build initial models
        self.rebuild_models()
    
    def rebuild_models(self):
        """Rebuild GARCH models and strategy"""
        print("\n🔧 Rebuilding models...")
        
        self.garch_model = GARCHVolatility(
            p=self.config['garch_p'],
            q=self.config['garch_q']
        )
        
        self.strategy = VolatilityStrategy(
            vol_percentile=self.config['vol_percentile'],
            rsi_oversold=self.config['rsi_oversold'],
            rsi_overbought=self.config['rsi_overbought'],
            min_momentum=self.config['min_momentum']
        )
        
        self.last_model_rebuild = datetime.now()
        print(f"✅ Models rebuilt at {self.last_model_rebuild.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def should_rebuild_models(self):
        """Check if it's time to rebuild models (daily)"""
        if self.last_model_rebuild is None:
            return True
        
        hours_since_rebuild = (datetime.now() - self.last_model_rebuild).total_seconds() / 3600
        return hours_since_rebuild >= 24
    
    def analyze_crypto(self, crypto):
        """Analyze a single cryptocurrency"""
        symbol = f"{crypto}USDT"
        
        try:
            # Fetch data
            df = self.data_fetcher.fetch_ohlcv(
                symbol=symbol,
                interval=self.config['interval'],
                days=self.config['lookback_days']
            )
            
            if df is None or len(df) < 100:
                return None
            
            # Calculate GARCH volatility
            df['volatility'] = self.garch_model.fit_predict(df['Close'].pct_change())
            df = df.dropna()
            
            if len(df) < 100:
                return None
            
            # Generate signals
            df_with_signals = self.strategy.generate_signals(df)
            
            # Get latest signal
            latest_row = df_with_signals.iloc[-1]
            
            # Get Roostoo price
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
                'returns': latest_row['returns']
            }
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error analyzing {crypto}: {e}")
            return None
    
    def execute_trade(self, analysis):
        """Execute a trade based on analysis"""
        crypto = analysis['crypto']
        signal = analysis['signal']
        confidence = analysis['confidence']
        price = analysis['price']
        
        # Check if we should trade
        if signal == -1:
            return False
        
        if confidence < self.min_confidence:
            print(f"⏸️  {crypto}: Confidence {confidence:.0%} < minimum {self.min_confidence:.0%}, skipping trade")
            return False
        
        # Get balance
        balance = self.roostoo_apis[crypto].get_balance()
        if not balance or not balance.get('Success'):
            print(f"❌ {crypto}: Could not fetch balance")
            return False
        
        wallet = balance.get('SpotWallet', {})
        
        # Calculate position size
        if signal == 1:  # BUY
            usd_balance = wallet.get('USD', {}).get('Free', 0)
            
            if usd_balance <= 0:
                print(f"⏸️  {crypto}: No USD balance available")
                return False
            
            # Calculate how much to buy
            position_usd = min(
                usd_balance * self.config['position_size_pct'],
                self.config['max_position_usd']
            )
            
            # Check minimum position size
            if position_usd < self.config.get('min_position_usd', 50):
                print(f"⏸️  {crypto}: Position size ${position_usd:.2f} < minimum ${self.config.get('min_position_usd', 50):.2f}")
                return False
            
            quantity = position_usd / price
            side = 'BUY'
            reason = f"Volatility signal (RSI: {analysis['rsi']:.1f}, Vol: {analysis['volatility']:.4f})"
            
        else:  # SELL
            crypto_balance = wallet.get(crypto, {}).get('Free', 0)
            
            if crypto_balance <= 0:
                print(f"⏸️  {crypto}: No {crypto} balance to sell")
                return False
            
            # Sell a portion
            quantity = crypto_balance * self.config['position_size_pct']
            side = 'SELL'
            reason = f"Volatility signal (RSI: {analysis['rsi']:.1f}, Vol: {analysis['volatility']:.4f})"
        
        # Execute order
        print(f"\n💸 Executing {side} order for {crypto}...")
        print(f"   Quantity: {quantity:.6f}")
        print(f"   Price: ${price:,.2f}")
        print(f"   Confidence: {confidence:.0%}")
        
        order = self.roostoo_apis[crypto].place_order(
            side=side,
            quantity=quantity,
            order_type='MARKET'
        )
        
        if order and order.get('Success'):
            order_detail = order.get('OrderDetail', {})
            self.logger.log_trade(
                crypto=crypto,
                action=side,
                quantity=order_detail.get('FilledQuantity', quantity),
                price=order_detail.get('FilledAverPrice', price),
                reason=reason,
                order_id=order_detail.get('OrderID'),
                success=True
            )
            return True
        else:
            self.logger.log_trade(
                crypto=crypto,
                action=side,
                quantity=quantity,
                price=price,
                reason=reason,
                success=False
            )
            return False
    
    def run_cycle(self):
        """Run a single trading cycle"""
        self.cycle_count += 1
        
        print("\n" + "="*80)
        print(f"🔄 CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Check if we need to rebuild models
        if self.should_rebuild_models():
            self.rebuild_models()
        
        # Analyze all cryptos
        analyses = {}
        for crypto in self.cryptos:
            print(f"\n📊 Analyzing {crypto}...")
            analysis = self.analyze_crypto(crypto)
            if analysis:
                analyses[crypto] = analysis
                
                signal_text = "BUY 🟢" if analysis['signal'] == 1 else "SELL 🔴" if analysis['signal'] == 0 else "HOLD ➖"
                print(f"   Signal: {signal_text}")
                print(f"   Confidence: {analysis['confidence']:.0%}")
                print(f"   Price: ${analysis['price']:,.2f}")
                print(f"   RSI: {analysis['rsi']:.1f}")
                print(f"   Volatility: {analysis['volatility']:.4f}")
        
        # Execute trades if auto-trading is enabled
        if self.auto_trade:
            print(f"\n💼 Auto-trading enabled - executing trades...")
            for crypto, analysis in analyses.items():
                if analysis['signal'] != -1 and analysis['confidence'] >= self.min_confidence:
                    self.execute_trade(analysis)
                    time.sleep(2)  # Small delay between trades
        else:
            print(f"\n⏸️  Auto-trading disabled - showing recommendations only")
            for crypto, analysis in analyses.items():
                if analysis['signal'] != -1 and analysis['confidence'] >= self.min_confidence:
                    action = "BUY" if analysis['signal'] == 1 else "SELL"
                    print(f"   💡 {action} {crypto} @ ${analysis['price']:,.2f} (confidence: {analysis['confidence']:.0%})")
        
        # Log cycle summary
        self.logger.log_cycle(self.cycle_count, analyses)
        
        print(f"\n✅ Cycle #{self.cycle_count} completed")
    
    def run_forever(self, cycle_interval_minutes=5):
        """Run bot forever with specified cycle interval"""
        print("\n" + "🚀"*40)
        print("24/7 AUTO-TRADING BOT STARTED")
        print("🚀"*40)
        print(f"\n⚙️  Configuration:")
        print(f"   Cryptos: {', '.join(self.cryptos)}")
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
                    # Run cycle
                    self.run_cycle()
                    
                    # Print summary every 10 cycles
                    if self.cycle_count % 10 == 0:
                        print(self.logger.get_summary())
                    
                    # Wait for next cycle
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
    print("🤖 ROOSTOO 24/7 AUTO-TRADING BOT")
    print("="*80)
    print("Multi-Crypto Volatility Strategy with Automatic Execution")
    print("\nFeatures:")
    print("  • Analyzes BTC, ETH, SOL every 5 minutes")
    print("  • Automatically executes trades based on GARCH volatility signals")
    print("  • Rebuilds models daily for optimal performance")
    print("  • Comprehensive trade logging and monitoring")
    print("="*80)
    
    # Configuration
    HORUS_API_KEY = "2458b11dad77377f2c13f1401776436cd9f386148220635cf399816355bc28c2"
    
    # Safety settings
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
    
    print("\n" + "="*80)
    print("FINAL CONFIGURATION:")
    print("="*80)
    print(f"Auto-trading: {'ENABLED ✅' if auto_trade else 'DISABLED ⏸️ (Recommendations only)'}")
    print(f"Min Confidence: {min_confidence:.0%}")
    print(f"Cycle Interval: {cycle_minutes} minutes")
    print("="*80)
    
    if auto_trade:
        print("\n⚠️  FINAL WARNING: Real trades will be executed!")
        final_confirm = input("Press ENTER to start, or Ctrl+C to cancel...")
    
    # Initialize and run bot
    bot = AutoTradingBot(
        horus_api_key=HORUS_API_KEY,
        auto_trade=auto_trade,
        min_confidence=min_confidence
    )
    
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