import requests
import time
import hmac
import hashlib
import math
from datetime import datetime

class RoostooRebalancingBot:
    def __init__(self, api_key, secret_key, initial_capital=10000, threshold=0.005):
        self.base_url = "https://mock-api.roostoo.com"
        self.api_key = api_key
        self.secret_key = secret_key
        self.initial_capital = initial_capital
        self.threshold = threshold
        
        # Target allocations
        self.targets = {
            'BTC/USD': 0.50,
            'ETH/USD': 0.30,
            'SOL/USD': 0.20
        }
        
        # Track if we've done initial allocation
        self.initialized = False
        self.trade_history = []
        
        # 24/7 mode settings
        self.error_count = 0
        self.max_consecutive_errors = 10
        self.last_successful_check = None
    
    def _get_timestamp(self):
        """Return a 13-digit millisecond timestamp as string."""
        return str(int(time.time() * 1000))
    
    def _get_signed_headers(self, payload={}):
        """Generate signed headers for authenticated endpoints."""
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }
        
        return headers, payload, total_params
    
    def get_exchange_info(self):
        """Get exchange trading rules."""
        url = f"{self.base_url}/v3/exchangeInfo"
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting exchange info: {e}")
            return None
    
    def get_ticker(self, pair=None):
        """Get current market prices."""
        url = f"{self.base_url}/v3/ticker"
        params = {'timestamp': self._get_timestamp()}
        
        if pair:
            params['pair'] = pair
        
        try:
            res = requests.get(url, params=params, timeout=30)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting ticker: {e}")
            return None
    
    def get_prices(self):
        """Fetch current prices for all target symbols."""
        ticker_data = self.get_ticker()
        
        if not ticker_data or not ticker_data.get('Success'):
            return None
        
        prices = {}
        data = ticker_data.get('Data', {})
        
        for pair in self.targets.keys():
            if pair in data:
                prices[pair] = data[pair]['LastPrice']
            else:
                print(f"⚠️  No price data for {pair}")
                return None
        
        return prices
    
    def get_balance(self):
        """Get current wallet balances."""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        
        try:
            res = requests.get(url, headers=headers, params=payload, timeout=30)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting balance: {e}")
            return None
    
    def get_portfolio_value(self):
        """Calculate total portfolio value from current balances and prices."""
        balance_data = self.get_balance()
        prices = self.get_prices()
        
        if not balance_data or not prices or not balance_data.get('Success'):
            return None, None, None
        
        # Use SpotWallet (not Wallet)
        wallet = balance_data.get('SpotWallet', {})
        
        # Extract coin symbols from pairs (e.g., BTC/USD -> BTC)
        holdings = {}
        for pair in self.targets.keys():
            coin = pair.split('/')[0]
            if coin in wallet:
                holdings[pair] = wallet[coin]['Free'] + wallet[coin]['Lock']
            else:
                holdings[pair] = 0
        
        # Calculate USD balance
        usd_balance = 0
        if 'USD' in wallet:
            usd_balance = wallet['USD']['Free'] + wallet['USD']['Lock']
        
        # Calculate total portfolio value
        total_value = usd_balance
        for pair, amount in holdings.items():
            total_value += amount * prices[pair]
        
        return total_value, holdings, usd_balance
    
    def place_order(self, pair, side, quantity, order_type='MARKET', price=None):
        """Place an order on Roostoo."""
        url = f"{self.base_url}/v3/place_order"
        
        payload = {
            'pair': pair,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        
        if order_type.upper() == 'LIMIT' and price:
            payload['price'] = str(price)
        
        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        
        try:
            res = requests.post(url, headers=headers, data=total_params, timeout=30)
            res.raise_for_status()
            result = res.json()
            
            if result.get('Success'):
                order_detail = result.get('OrderDetail', {})
                self.log_trade(
                    side,
                    pair,
                    order_detail.get('FilledQuantity', quantity),
                    order_detail.get('FilledAverPrice', 0),
                    f"{order_type} order",
                    order_detail.get('OrderID')
                )
            
            return result
        except requests.exceptions.RequestException as e:
            print(f"❌ Error placing order: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return None
    
    def check_existing_holdings(self):
        """Check if we already have crypto holdings and skip initial allocation if so."""
        print("🔍 Checking for existing holdings...")
        
        total_value, holdings, usd_balance = self.get_portfolio_value()
        
        if not total_value:
            print("❌ Failed to get portfolio value")
            return False
        
        # Check if we have any significant crypto holdings
        has_crypto = False
        for pair, amount in holdings.items():
            coin = pair.split('/')[0]
            if amount > 0:
                has_crypto = True
                print(f"  💎 Found {coin}: {amount}")
        
        if has_crypto:
            print(f"\n✅ Existing holdings detected!")
            print(f"   Total Portfolio Value: ${total_value:,.2f}")
            print(f"   USD Balance: ${usd_balance:,.2f}")
            print(f"\n⚡ Skipping initial allocation, will start rebalancing...")
            self.initialized = True
            return True
        else:
            print(f"  📭 No existing crypto holdings found")
            print(f"  💵 USD Balance: ${usd_balance:,.2f}")
            return False
    
    def initial_allocation(self):
        """Execute initial portfolio allocation."""
        print("\n🚀 Starting initial portfolio allocation...")
        
        # Get exchange info first to know the rules
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            print("❌ Failed to get exchange info")
            return False
        
        trade_pairs = exchange_info.get('TradePairs', {})
        
        prices = self.get_prices()
        if not prices:
            print("❌ Failed to get prices. Aborting.")
            return False
        
        balance_data = self.get_balance()
        if not balance_data or not balance_data.get('Success'):
            print("❌ Failed to get balance. Aborting.")
            return False
        
        wallet = balance_data.get('SpotWallet', {})
        available_usd = wallet.get('USD', {}).get('Free', 0)
        
        if available_usd <= 0:
            print(f"❌ No USD available (Balance: ${available_usd}). Aborting.")
            return False
        
        print(f"💵 Available USD: ${available_usd:,.2f}")
        print(f"\n📊 Target Allocation:")
        
        for pair, target_pct in self.targets.items():
            print(f"  {pair}: {target_pct*100:.0f}%")
        
        print("\n🔄 Executing initial buys...")
        
        # Execute market buy orders for each asset
        for pair, target_pct in self.targets.items():
            amount_to_invest = available_usd * target_pct
            quantity = amount_to_invest / prices[pair]
            
            # Get the EXACT precision required by the exchange
            pair_info = trade_pairs.get(pair, {})
            amount_precision = pair_info.get('AmountPrecision', 2)
            mini_order = pair_info.get('MiniOrder', 1.0)
            
            # Use floor instead of round to avoid going over precision
            multiplier = 10 ** amount_precision
            quantity = math.floor(quantity * multiplier) / multiplier
            
            # Check minimum order value
            order_value = quantity * prices[pair]
            if order_value < mini_order:
                print(f"  ⚠️  Skipping {pair} - order value ${order_value:.2f} < minimum ${mini_order}")
                continue
            
            if quantity <= 0:
                print(f"  ⚠️  Skipping {pair} - quantity too small")
                continue
            
            print(f"\n  📝 Buying {quantity:.{amount_precision}f} {pair.split('/')[0]} (${amount_to_invest:.2f})")
            print(f"      Precision: {amount_precision} decimals, Mini Order: ${mini_order}")
            
            result = self.place_order(pair, 'BUY', quantity, 'MARKET')
            
            if result and result.get('Success'):
                print(f"  ✅ Order successful")
            else:
                err_msg = result.get('ErrMsg') if result else 'Unknown error'
                print(f"  ❌ Order failed: {err_msg}")
            
            time.sleep(0.5)  # Small delay between orders
        
        self.initialized = True
        print("\n✅ Initial allocation complete!")
        return True
    
    def get_current_allocations(self):
        """Calculate current allocation percentages."""
        total_value, holdings, _ = self.get_portfolio_value()
        prices = self.get_prices()
        
        if not total_value or not prices or total_value == 0:
            return None, None, None
        
        allocations = {}
        for pair, amount in holdings.items():
            value = amount * prices[pair]
            allocations[pair] = value / total_value
        
        return allocations, total_value, holdings
    
    def check_and_rebalance(self):
        """Check if rebalancing is needed and execute trades."""
        try:
            allocations, portfolio_value, holdings = self.get_current_allocations()
            prices = self.get_prices()
            
            if not allocations or not prices:
                print("❌ Failed to get portfolio data. Skipping this cycle.")
                self.error_count += 1
                return
            
            # Reset error count on successful data fetch
            self.error_count = 0
            self.last_successful_check = datetime.now()
            
            print(f"\n📊 Portfolio Value: ${portfolio_value:,.2f}")
            print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nCurrent Allocations:")
            
            needs_rebalancing = False
            rebalance_actions = []
            
            for pair, current_pct in allocations.items():
                target_pct = self.targets[pair]
                diff = current_pct - target_pct
                deviation = abs(diff)
                status = "🔴" if deviation > self.threshold else "🟢"
                
                print(f"  {status} {pair}: {current_pct*100:.2f}% (target: {target_pct*100:.0f}%, diff: {diff*100:+.2f}%)")
                
                if deviation > self.threshold:
                    needs_rebalancing = True
                    
                    # Calculate how much to buy or sell
                    target_value = portfolio_value * target_pct
                    current_value = holdings[pair] * prices[pair]
                    value_diff = target_value - current_value
                    quantity_diff = value_diff / prices[pair]
                    
                    rebalance_actions.append({
                        'pair': pair,
                        'action': 'BUY' if quantity_diff > 0 else 'SELL',
                        'quantity': abs(quantity_diff)
                    })
            
            if needs_rebalancing:
                print("\n⚖️  Rebalancing triggered!")
                self.execute_rebalance(rebalance_actions)
            else:
                print("✅ Portfolio is balanced.")
                
        except Exception as e:
            print(f"❌ Error in check_and_rebalance: {e}")
            self.error_count += 1
    
    def execute_rebalance(self, actions):
        """Execute rebalancing trades."""
        print("\n🔄 Executing rebalancing trades...")
        
        # Get exchange info
        exchange_info = self.get_exchange_info()
        if not exchange_info:
            print("❌ Failed to get exchange info")
            return
        
        trade_pairs = exchange_info.get('TradePairs', {})
        
        for action in actions:
            pair = action['pair']
            side = action['action']
            quantity = action['quantity']
            
            # Get exact precision from exchange
            pair_info = trade_pairs.get(pair, {})
            amount_precision = pair_info.get('AmountPrecision', 2)
            mini_order = pair_info.get('MiniOrder', 1.0)
            
            # Floor the quantity to exact precision
            multiplier = 10 ** amount_precision
            quantity = math.floor(quantity * multiplier) / multiplier
            
            if quantity <= 0:
                print(f"  ⚠️  Skipping {pair} - quantity too small after rounding")
                continue
            
            # Check minimum order value
            prices = self.get_prices()
            if prices:
                order_value = quantity * prices[pair]
                if order_value < mini_order:
                    print(f"  ⚠️  Skipping {pair} - order value ${order_value:.2f} < minimum ${mini_order}")
                    continue
            
            print(f"\n  📝 {side} {quantity:.{amount_precision}f} {pair.split('/')[0]}")
            
            result = self.place_order(pair, side, quantity, 'MARKET')
            
            if result and result.get('Success'):
                print(f"  ✅ Order successful")
            else:
                print(f"  ❌ Order failed: {result.get('ErrMsg') if result else 'Unknown error'}")
            
            time.sleep(0.5)  # Small delay between orders
    
    def log_trade(self, action, pair, quantity, price, reason, order_id=None):
        """Log trade to history."""
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'pair': pair,
            'quantity': quantity,
            'price': price,
            'total': quantity * price,
            'reason': reason,
            'order_id': order_id
        }
        self.trade_history.append(trade)
        
        emoji = "🟢" if action == "BUY" else "🔴"
        print(f"  {emoji} {action} {quantity:.6f} {pair} @ ${price:.2f} (Order #{order_id})")
    
    def print_summary(self):
        """Print current portfolio summary."""
        allocations, portfolio_value, holdings = self.get_current_allocations()
        prices = self.get_prices()
        
        if not allocations or not prices:
            print("❌ Cannot generate summary")
            return
        
        # Calculate P&L based on initial capital
        total_value, _, _ = self.get_portfolio_value()
        pnl = total_value - self.initial_capital
        pnl_pct = (pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        print("\n" + "="*60)
        print(f"💰 PORTFOLIO SUMMARY")
        print("="*60)
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Current Value:    ${total_value:,.2f}")
        print(f"P&L:             ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        print(f"Uptime:          {datetime.now() - self.start_time if hasattr(self, 'start_time') else 'N/A'}")
        
        print(f"\n📈 Holdings:")
        for pair, amount in holdings.items():
            value = amount * prices[pair]
            pct = (value / total_value) * 100 if total_value > 0 else 0
            coin = pair.split('/')[0]
            print(f"  {coin}: {amount:.8f} @ ${prices[pair]:,.2f} = ${value:,.2f} ({pct:.1f}%)")
        
        print(f"\n📊 Total Trades Executed: {len(self.trade_history)}")
        print(f"🔄 Last successful check: {self.last_successful_check.strftime('%Y-%m-%d %H:%M:%S') if self.last_successful_check else 'Never'}")
        
        if self.trade_history:
            print(f"\n📜 Recent Trades (last 5):")
            for i, trade in enumerate(self.trade_history[-5:], 1):
                print(f"  {i}. {trade['timestamp'].strftime('%H:%M:%S')} - "
                      f"{trade['action']} {trade['quantity']:.6f} {trade['pair']} "
                      f"@ ${trade['price']:.2f}")
        
        print("="*60)
    
    def run_forever(self, check_interval=60, summary_interval=3600):
        """Run the bot continuously (24/7 mode)."""
        self.start_time = datetime.now()
        
        print("="*60)
        print("🤖 CRYPTO REBALANCING BOT - 24/7 MODE")
        print("="*60)
        print(f"Strategy: 50% BTC / 30% ETH / 20% SOL")
        print(f"Rebalancing Threshold: {self.threshold*100}%")
        print(f"Check Interval: {check_interval} seconds")
        print(f"Summary Interval: {summary_interval} seconds")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
        # Check server connection
        print("🔌 Checking server connection...")
        ticker = self.get_ticker()
        if ticker and ticker.get('Success'):
            print(f"✅ Connected to Roostoo API (Server Time: {ticker.get('ServerTime')})")
        else:
            print("❌ Failed to connect to Roostoo API")
            print("⏳ Will retry in 60 seconds...")
            time.sleep(60)
        
        # Check for existing holdings first
        if self.check_existing_holdings():
            # We have existing holdings, skip initial allocation
            pass
        else:
            # No existing holdings, do initial allocation
            retry_count = 0
            while not self.initial_allocation() and retry_count < 3:
                retry_count += 1
                print(f"⏳ Retrying initial allocation ({retry_count}/3)...")
                time.sleep(30)
            
            if retry_count >= 3:
                print("❌ Failed to complete initial allocation after 3 attempts")
                print("⚠️  Bot will continue and try to rebalance existing holdings")
                self.initialized = True
        
        # Main loop
        iteration = 1
        last_summary = time.time()
        
        print("\n🚀 Starting 24/7 rebalancing loop...")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Check if we've had too many consecutive errors
                if self.error_count >= self.max_consecutive_errors:
                    print(f"\n⚠️  Too many consecutive errors ({self.error_count})")
                    print("⏳ Pausing for 5 minutes before retry...")
                    time.sleep(300)
                    self.error_count = 0
                
                print(f"\n{'='*60}")
                print(f"🔄 Check #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                uptime = datetime.now() - self.start_time
                print(f"⏱️  Uptime: {uptime}")
                print('='*60)
                
                self.check_and_rebalance()
                
                # Print summary periodically
                if time.time() - last_summary >= summary_interval:
                    self.print_summary()
                    last_summary = time.time()
                
                iteration += 1
                
                print(f"\n⏱️  Waiting {check_interval}s until next check...")
                print(f"   (Next check at {(datetime.now() + timedelta(seconds=check_interval)).strftime('%H:%M:%S')})")
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Bot stopped by user")
            self.print_summary()
            print("\n👋 Goodbye!")
    
    def run(self, check_interval=60, duration=None):
        """
        Run the bot.
        
        Args:
            check_interval: Seconds between rebalance checks
            duration: Run duration in seconds (None = run forever)
        """
        if duration is None:
            # 24/7 mode
            self.run_forever(check_interval=check_interval)
        else:
            # Demo mode with duration
            self.run_demo(check_interval=check_interval, duration=duration)
    
    def run_demo(self, check_interval=60, duration=3600):
        """Run the bot for a specified duration (demo mode)."""
        self.start_time = datetime.now()
        
        print("="*60)
        print("🤖 CRYPTO REBALANCING BOT - DEMO MODE")
        print("="*60)
        print(f"Strategy: 50% BTC / 30% ETH / 20% SOL")
        print(f"Rebalancing Threshold: {self.threshold*100}%")
        print(f"Check Interval: {check_interval} seconds")
        print(f"Run Duration: {duration} seconds ({duration/60:.1f} minutes)")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print("="*60 + "\n")
        
        # Check server connection
        print("🔌 Checking server connection...")
        ticker = self.get_ticker()
        if ticker and ticker.get('Success'):
            print(f"✅ Connected to Roostoo API (Server Time: {ticker.get('ServerTime')})")
        else:
            print("❌ Failed to connect to Roostoo API")
            return
        
        # Check for existing holdings first
        if self.check_existing_holdings():
            # We have existing holdings, skip initial allocation
            pass
        else:
            # No existing holdings, do initial allocation
            if not self.initial_allocation():
                print("❌ Initial allocation failed. Aborting.")
                return
        
        # Main rebalancing loop
        start_time = time.time()
        iteration = 1
        
        try:
            while time.time() - start_time < duration:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                print(f"\n{'='*60}")
                print(f"🔄 Check #{iteration} - Elapsed: {elapsed/60:.1f}m / Remaining: {remaining/60:.1f}m")
                print('='*60)
                
                self.check_and_rebalance()
                
                if remaining > check_interval:
                    print(f"\n⏱️  Waiting {check_interval}s until next check...")
                    time.sleep(check_interval)
                else:
                    print(f"\n⏱️  Final check complete. Waiting {remaining:.0f}s until end...")
                    time.sleep(max(0, remaining))
                    break
                
                iteration += 1
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Bot stopped by user")
        
        self.print_summary()


# Import timedelta for 24/7 mode
from datetime import timedelta

# Run the bot
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual Roostoo API credentials
    API_KEY = "olDFYeqSIQgk5phJgY4lWKNNaGpJiWnnecDeIWitZLV23DtnP15lGJJbKi1BsOPD"
    SECRET_KEY = "Azfl3pdnzckPEYVR3pHJf3CzPtESK3bjkOUKlwqZZOlEThheiqqlZNlPTPg1sRYM"
    
    # Create the bot
    bot = RoostooRebalancingBot(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        initial_capital=50000,  # Adjust based on your actual starting capital
        threshold=0.005          # 0.5% rebalancing threshold
    )
    
    # ============================================
    # CHOOSE YOUR MODE:
    # ============================================
    
    # 1. DEMO MODE (for testing - 5 minutes):
    # bot.run(check_interval=30, duration=300)
    
    # 2. 24/7 MODE (for hackathon submission):
    bot.run(check_interval=60, duration=None)  # duration=None means run forever
    
    # 3. Custom demo (30 minutes):
    # bot.run(check_interval=60, duration=1800)
