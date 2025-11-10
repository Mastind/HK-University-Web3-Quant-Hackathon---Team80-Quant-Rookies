# 🤖 ROOSTOO 24/7 Auto-Trading Bot

A sophisticated multi-currency algorithmic trading bot that analyzes 6 major cryptocurrencies using individual GARCH volatility models and automatic portfolio rebalancing.

## 🚀 Features

### 📊 Multi-Currency Analysis
- **6 Supported Cryptocurrencies**: BTC, ETH, SOL, BNB, XRP, ADA
- **Individual GARCH Models**: Each crypto has its own volatility model built from its unique historical data
- **Real-time Data**: Fetches live market data from Horus API
- **24/7 Operation**: Continuous monitoring and automated trading

### ⚡ Advanced Trading Strategies
- **GARCH Volatility Forecasting**: Predicts price volatility using Generalized Autoregressive Conditional Heteroskedasticity models
- **RSI Momentum Signals**: Combines volatility with Relative Strength Index for entry/exit points
- **Confidence-Based Execution**: Only executes trades meeting minimum confidence thresholds
- **Risk Management**: Position sizing and maximum trade limits

### 🔄 Smart Portfolio Management
- **Automatic Rebalancing**: Maintains target portfolio allocations (BTC 30%, ETH 25%, SOL 20%, ADA 10%, BNB 10%, XRP 5%)
- **1% Threshold**: Triggers rebalancing when any asset deviates more than 1% from target
- **Rate Limit Compliance**: Respects 1 trade per minute restriction
- **Retry Logic**: Automatic retry with 15-second delays for failed trades

## 🛠 Technical Architecture

### Core Components
- **Roostoo API Integration**: Handles order execution and balance management
- **Horus Data Fetcher**: Retrieves historical data for volatility modeling
- **GARCH Volatility Engine**: Individual models for each cryptocurrency
- **Portfolio Rebalancer**: Automated allocation management
- **Trade Logger**: Comprehensive transaction tracking and reporting

### Risk Controls
- Minimum confidence thresholds (60-80%)
- Position size limits (configurable)
- Maximum trade value restrictions
- Automatic stop-loss through volatility signals

## 📋 Prerequisites

### Python Requirements
```bash
Python 3.8+
Required packages:
- pandas
- numpy
- requests
- arch
- statsmodels
- scipy
- hmac
- hashlib
```

### API Keys
- **Roostoo Exchange**: Trading and order execution
- **Horus API**: Historical price data (included in code)

## ⚙️ Installation

1. **Clone Repository**
```bash
git clone https://github.com/Mastind/HK-University-Web3-Quant-Hackathon---Team80-Quant-Rookies.git
cd HK-University-Web3-Quant-Hackathon---Team80-Quant-Rookies
```

2. **Install Dependencies**
```bash
pip install pandas numpy requests statsmodels scipy arch
```

3. **Configure Trading Parameters**
   - Edit the main script to adjust:
   - Minimum confidence levels
   - Position sizes
   - Trading intervals
   - Rebalancing thresholds

4. **Run the Bot**
```bash
python multi_volatility_bot_with_rebalancing.py
```

## 🎯 Configuration

### Trading Parameters
```python
config = {
    'interval': '15m',           # Analysis timeframe
    'lookback_days': 14,         # Historical data period
    'vol_percentile': 0.65,      # Volatility threshold
    'position_size_pct': 0.50,   # % of balance per trade
    'max_position_usd': 30000,   # Maximum trade size
    'min_confidence': 0.65,      # Minimum signal confidence
}
```

### Portfolio Allocation Targets
| Cryptocurrency | Target Allocation |
|----------------|-------------------|
| BTC            | 30%               |
| ETH            | 25%               |
| SOL            | 20%               |
| ADA            | 10%               |
| BNB            | 10%               |
| XRP            | 5%                |

## 📈 How It Works

### 1. Data Collection & Analysis
- Fetches historical price data for each cryptocurrency
- Builds individual GARCH volatility models
- Calculates RSI and momentum indicators

### 2. Signal Generation
- Identifies volatility expansion opportunities
- Combines multiple technical indicators
- Assigns confidence scores to each signal

### 3. Portfolio Management
- Monitors current allocations vs targets
- Executes rebalancing trades when deviations exceed 1%
- Respects rate limits and retries failed transactions

### 4. Risk Management
- Position sizing based on portfolio percentage
- Maximum trade limits to control exposure
- Confidence-based trade filtering

## 🚨 Important Disclaimers

### ⚠️ Risk Warning
- **This is experimental software** for educational purposes
- **Cryptocurrency trading involves substantial risk**
- **Past performance does not guarantee future results**
- **You can lose your entire investment**

### 🔒 Security Notes
- API keys are hardcoded for demonstration (use environment variables in production)
- Always test with small amounts first
- Monitor the bot regularly, especially during high volatility

## 📊 Performance Monitoring

The bot provides comprehensive logging:
- Real-time trade execution updates
- Portfolio allocation reports
- Success/failure rate tracking
- Performance summaries every 10 cycles

## 🛠 Customization

### Adding New Cryptocurrencies
1. Update `trading_rules` in RoostooAPI class
2. Add to `symbol_map` in HorusDataFetcher
3. Include in `cryptos` list and target allocations

### Modifying Trading Strategy
- Adjust GARCH model parameters (p, q values)
- Modify RSI thresholds and volatility percentiles
- Change position sizing logic

## 🐛 Troubleshooting

### Common Issues
- **API Rate Limits**: Bot automatically handles 1 trade/minute restriction
- **Data Fetching Errors**: Retry logic with fallback mechanisms
- **Insufficient Balance**: Position sizing adjusts automatically
- **Network Issues**: Built-in reconnection and retry logic

### Logs and Debugging
- Detailed console output for all operations
- Trade history with timestamps and order IDs
- Error tracking with full stack traces

## 📄 License

This project is for educational purposes as part of a hackathon submission. Use at your own risk.

## 🤝 Contributing

This is a hackathon project. For questions or issues, please open a GitHub issue.

---

**Built for Web3 Hackathon** | *Use responsibly and always test with small amounts first*
