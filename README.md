# AI Hedge Fund 🤖📈

An advanced AI-powered hedge fund system developed by Jac, leveraging Google's Gemini AI for real-time market analysis and trading decisions.

## Overview

This system uses a multi-agent architecture powered by Google Gemini AI to analyze stocks and make trading decisions. Each agent specializes in different aspects of market analysis:

- Technical Analysis Agent: Evaluates price patterns, trends, and technical indicators
- Fundamental Analysis Agent: Assesses financial metrics and company health
- Sentiment Analysis Agent: Analyzes market sentiment and news impact
- Risk Management Agent: Controls position sizing and risk exposure
- Portfolio Management Agent: Makes final trading decisions based on all signals

## Features

- Real-time stock analysis using Google Gemini AI
- Comprehensive web research and data gathering
- Multi-timeframe analysis (intraday, daily, weekly)
- Advanced technical indicators with confidence levels
- Sentiment analysis with news impact assessment
- Institutional ownership and options market analysis
- Risk-adjusted position sizing
- Well-structured analysis output in text format
- Detailed reasoning for all trading decisions

## Setup Guide

### Prerequisites
- Python 3.8 or higher
- Git
- Google Gemini API key

### Step 1: Clone the Repository
```bash
git clone ai-hedge-fund
cd ai-hedge-fund
```

### Step 2: Set Up Python Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
1. Create a `.env` file in the root directory
2. Add your Google Gemini API key:
```
GOOGLE_GEMINI_API_KEY=your-api-key-here
```

### Step 5: Run the System
```bash
# Basic run
python src/main.py

# Show detailed reasoning
python src/main.py --show-reasoning
```

## Usage

1. When prompted, enter a stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
2. The system will:
   - Gather real-time market data using Gemini AI
   - Perform comprehensive analysis across all agents
   - Generate a detailed analysis report
   - Provide a final trading decision with confidence level
   - Save results to the `outputs` directory

### Example Output Structure
```
=== Investment Analysis Report for TICKER ===
Generated on: YYYY-MM-DD_HHMMSS

=== OVERALL DECISION ===
DECISION: BUY/SELL/HOLD
CONFIDENCE: XX%
POSITION SIZE: XX%

SUPPORTING SIGNALS:
- Technical Analysis
- Fundamental Analysis
- Sentiment Analysis
- Risk Assessment

=== DETAILED ANALYSIS ===
[Comprehensive analysis from each agent]
```

## Advanced Features

### Technical Analysis
- Multi-timeframe trend analysis (short-term, medium-term, long-term)
- Advanced momentum indicators (RSI, MACD, Stochastic)
- Volume analysis with confidence levels
- Support/resistance identification
- Pattern recognition and trend strength assessment
- Moving average crossovers and signals
- Bollinger Bands analysis
- Volatility measurements

### Fundamental Analysis
- Financial metrics evaluation
  - Profitability ratios
  - Growth metrics
  - Valuation multiples
- Balance sheet analysis
- Cash flow assessment
- Earnings quality evaluation
- Industry comparison
- Market share analysis
- Competitive positioning

### Sentiment Analysis
- News impact assessment
  - Real-time news analysis
  - Social media sentiment
  - Market buzz tracking
- Market sentiment evaluation
- Institutional ownership tracking
  - Ownership changes
  - Major holder positions
- Options market sentiment
  - Put/call ratios
  - Implied volatility
  - Options flow analysis

### Risk Management
- Dynamic position sizing
- Risk-adjusted returns calculation
- Portfolio exposure control
- Stop-loss management
- Volatility-based adjustments
- Correlation analysis
- Drawdown protection
- Risk factor decomposition

## Disclaimer

This software is for educational and research purposes only. It is not intended for actual trading. Always consult with financial professionals before making investment decisions.

## License

MIT License - See LICENSE file for details

## Author

Created and maintained by Jac.

For questions or support, please reach out to the development team.
