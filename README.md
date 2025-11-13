# 🤖 ROOSTOO 24/7 Auto-Trading Bot

A sophisticated multi-cryptocurrency trading bot that automatically analyzes and executes trades on the Roostoo exchange using GARCH volatility models and technical indicators.

## 📊 Features

### Core Capabilities
- **24/7 Automated Trading** - Runs continuously with configurable cycle intervals
- **Multi-Crypto Support** - Trades BTC, ETH, SOL, BNB, XRP, ADA simultaneously
- **Smart Volatility Strategy** - Uses GARCH models to identify trading opportunities
- **Risk Management** - Built-in take-profit (2%) and stop-loss (1%) mechanisms
- **Real Order Tracking** - Queries actual Roostoo order history for accurate position tracking

### Technical Analysis
- **GARCH Volatility Models** - Advanced volatility forecasting rebuilt daily
- **RSI Indicators** - Momentum analysis with overbought/oversold levels
- **Confidence Scoring** - Probability-based trade signals
- **Multi-timeframe Analysis** - 15-minute to daily data aggregation

### Risk Controls
- **Position Sizing** - Configurable position limits (max $10,000 per trade)
- **Buy Cooldowns** - 10-minute minimum between buy signals per currency
- **Balance Protection** - Automatic balance checks and quantity rounding
- **Safe Trading** - Optional demo mode with recommendations only

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy requests arch scipy statsmodels
```

### Installation
1. Clone or download the bot script
2. Ensure you have Python 3.7+ installed
3. Install required dependencies

### Configuration
Run the bot and configure through the interactive setup:
```bash
python volatility_bot_multi_currency_nonstop.py
```
Don't forget to use your own API for roostoo and horus!

You'll be prompted for:
- **Auto-trading**: Enable/disable actual trade execution
- **Confidence Level**: Minimum signal confidence (60-80%)
- **Cycle Interval**: Analysis frequency (1-60 minutes)

## ⚙️ Configuration Options

### Trading Parameters
| Setting | Default | Description |
|---------|---------|-------------|
| Position Size | 60% | Percentage of balance per trade |
| Max Position | $10,000 | Maximum USD per trade |
| Take Profit | +2% | Automatic profit-taking |
| Stop Loss | -1% | Automatic loss protection |
| Buy Cooldown | 10min | Minimum time between buys |
| Min Confidence | 65% | Required signal strength |

### Technical Settings
| Setting | Default | Description |
|---------|---------|-------------|
| GARCH (p,q) | (2,3) | Volatility model parameters |
| RSI Oversold | 40 | Buy signal threshold |
| RSI Overbought | 60 | Sell signal threshold |
| Volatility Percentile | 65% | Minimum volatility threshold |
| Data Lookback | 14 days | Historical data period |

## 🔧 How It Works

### Trading Cycle (Every 5 Minutes)
1. **Data Collection** - Fetches latest price data from Horus API
2. **Volatility Analysis** - Calculates GARCH volatility forecasts
3. **Signal Generation** - Identifies buy/sell opportunities using RSI and volatility
4. **Risk Assessment** - Checks take-profit/stop-loss conditions based on actual order history
5. **Order Execution** - Places market orders on Roostoo exchange (if auto-trading enabled)
6. **Logging** - Records all activity for monitoring and analysis

### Model Rebuilding
- GARCH models are automatically rebuilt every 24 hours
- Ensures optimal performance with changing market conditions
- Uses latest 14 days of price data for calibration

## 📈 Strategy Logic

### Buy Signals
- High volatility conditions (above 65th percentile)
- RSI below 40 (oversold) with positive momentum
- Minimum 65% confidence level
- Respects 10-minute buy cooldown per currency

### Sell Signals
- High volatility with RSI above 60 (overbought) and negative momentum
- Take-profit triggered at +2% from last buy price
- Stop-loss triggered at -1% from last buy price
- Based on actual filled order prices from Roostoo

## 🛡️ Risk Management

### Safety Features
- **Balance Protection**: Never risks more than configured limits
- **Order Validation**: Automatic quantity rounding and validation
- **Error Handling**: Comprehensive exception handling with retries
- **API Resilience**: Robust API integration with timeout handling

### Monitoring
- Real-time trade logging with timestamps
- Cycle-by-cycle performance summary
- Position tracking with P/L calculations
- Comprehensive error reporting

## 📋 Output Example

```
🔄 CYCLE #45 - 2024-01-15 14:05:00
========================================

💰 Current Positions:
   BTC: Buy @ $42,150.25, Current @ $42,890.50 (+1.76%)
   ETH: Buy @ $2,520.80, Current @ $2,480.25 (-1.61%)

📊 Analyzing BTC...
   Signal: HOLD ➖
   Confidence: 58%
   Price: $42,890.50
   RSI: 52.1
   Volatility: 0.0234

💼 Auto-trading enabled - executing trades...
✅ [14:05:03] SELL 0.015000 ETH @ $2,480.25 - Stop-loss triggered
```

## ⚠️ Important Notes

### Exchange Compatibility
- Designed specifically for Roostoo Exchange API
- Uses mock API endpoints for testing
- Real trading requires valid API credentials

### Risk Disclaimer
> **Warning**: This is automated trading software that executes real financial transactions. Use at your own risk. Cryptocurrency trading involves substantial risk of loss. Always test with small amounts and understand the code before using with real funds.

### Development Status
- Currently uses mock Roostoo API endpoints
- Horus API integration for market data
- Production readiness requires exchange certification

## 🔄 Customization

### Adding New Cryptocurrencies
1. Update `cryptos` list in `AutoTradingBot.__init__()`
2. Add trading rules in `RoostooAPI.trading_rules`
3. Update symbol mapping in `HorusDataFetcher.symbol_map`

### Strategy Modification
- Adjust GARCH parameters in `GARCHVolatility`
- Modify signal logic in `VolatilityStrategy.generate_signals()`
- Update risk parameters in bot configuration

## 🐛 Troubleshooting

### Common Issues
- **"arch package not found"**: Run `pip install arch`
- **API connection errors**: Check internet connection and API status
- **Balance errors**: Verify API keys have trading permissions
- **Data quality issues**: Horus API may have occasional gaps

### Logs and Monitoring
- All trades logged with timestamps and order IDs
- Cycle summaries show signal confidence levels
- Regular performance summaries every 10 cycles

## 📄 License

This project is for educational and demonstration purposes. Use commercially at your own risk.

---

**Happy Trading!** 🚀📈

*For questions or issues, please review the code comments and configuration options carefully before seeking support.*
