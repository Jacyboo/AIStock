from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd

from main import run_hedge_fund
from tools.manual_input import get_price_data

class Backtester:
    def __init__(self, agent, ticker, start_date, end_date, initial_capital, manual_data):
        self.agent = agent
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.manual_data = manual_data
        self.portfolio = {"cash": initial_capital, "stock": 0}
        self.portfolio_values = []

    def parse_action(self, agent_output):
        try:
            # Expect JSON output from agent
            import json
            decision = json.loads(agent_output)
            return decision["action"], decision["quantity"]
        except:
            print(f"Error parsing action: {agent_output}")
            return "hold", 0

    def execute_trade(self, action, quantity, current_price):
        """Validate and execute trades based on portfolio constraints"""
        if action == "buy" and quantity > 0:
            cost = quantity * current_price
            if cost <= self.portfolio["cash"]:
                self.portfolio["stock"] += quantity
                self.portfolio["cash"] -= cost
                return quantity
            else:
                # Calculate maximum affordable quantity
                max_quantity = self.portfolio["cash"] // current_price
                if max_quantity > 0:
                    self.portfolio["stock"] += max_quantity
                    self.portfolio["cash"] -= max_quantity * current_price
                    return max_quantity
                return 0
        elif action == "sell" and quantity > 0:
            quantity = min(quantity, self.portfolio["stock"])
            if quantity > 0:
                self.portfolio["cash"] += quantity * current_price
                self.portfolio["stock"] -= quantity
                return quantity
            return 0
        return 0

    def run_backtest(self):
        dates = pd.date_range(self.start_date, self.end_date, freq="B")

        print("\nStarting backtest...")
        print(f"{'Date':<12} {'Ticker':<6} {'Action':<6} {'Quantity':>8} {'Price':>8} {'Cash':>12} {'Stock':>8} {'Total Value':>12}")
        print("-" * 100)

        for current_date in dates:
            lookback_start = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            current_date_str = current_date.strftime("%Y-%m-%d")

            agent_output = self.agent(
                ticker=self.ticker,
                start_date=lookback_start,
                end_date=current_date_str,
                portfolio=self.portfolio,
                manual_data=self.manual_data
            )

            action, quantity = self.parse_action(agent_output)
            df = get_price_data(self.manual_data["prices"], lookback_start, current_date_str)
            if df.empty:
                print(f"No price data available for {current_date_str}, skipping...")
                continue
            current_price = df.iloc[-1]['close']

            # Execute the trade with validation
            executed_quantity = self.execute_trade(action, quantity, current_price)

            # Update total portfolio value
            total_value = self.portfolio["cash"] + self.portfolio["stock"] * current_price
            self.portfolio["portfolio_value"] = total_value

            # Log the current state with executed quantity
            print(
                f"{current_date.strftime('%Y-%m-%d'):<12} {self.ticker:<6} {action:<6} {executed_quantity:>8} {current_price:>8.2f} "
                f"{self.portfolio['cash']:>12.2f} {self.portfolio['stock']:>8} {total_value:>12.2f}"
            )

            # Record the portfolio value
            self.portfolio_values.append(
                {"Date": current_date, "Portfolio Value": total_value}
            )

    def analyze_performance(self):
        """Calculate and display performance metrics"""
        if not self.portfolio_values:
            print("No portfolio values to analyze")
            return None

        df = pd.DataFrame(self.portfolio_values)
        df.set_index('Date', inplace=True)

        # Calculate returns
        df['Returns'] = df['Portfolio Value'].pct_change()

        # Calculate metrics
        total_return = (df['Portfolio Value'].iloc[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df['Returns'].std() * (252 ** 0.5)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = (df['Portfolio Value'] / df['Portfolio Value'].cummax() - 1).min()

        print("\nPerformance Metrics:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Annualized Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")

        return df

### Run the Backtest #####
if __name__ == "__main__":
    import argparse
    import json
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run backtesting simulation')
    parser.add_argument('--ticker', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date in YYYY-MM-DD format')
    parser.add_argument('--start_date', type=str, default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'), help='Start date in YYYY-MM-DD format')
    parser.add_argument('--initial_capital', type=float, default=100000, help='Initial capital amount (default: 100000)')
    parser.add_argument('--manual_data', type=str, required=True, help='Path to JSON file containing manual financial data')

    args = parser.parse_args()

    # Load manual data
    try:
        with open(args.manual_data, 'r') as f:
            manual_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Manual data file not found: {args.manual_data}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in manual data file: {args.manual_data}")

    # Create an instance of Backtester
    backtester = Backtester(
        agent=run_hedge_fund,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        manual_data=manual_data
    )

    # Run the backtesting process
    backtester.run_backtest()
    performance_df = backtester.analyze_performance()
